import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import faiss
import json
import os
import re
import copy
from torch.optim import AdamW
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import t
from scipy.optimize import linear_sum_assignment
from transformers import get_linear_schedule_with_warmup
from methods.ligd.model import Bert
from methods.ligd.utils_data import NIDData, NIDDataset, MemoryBank, fill_memory_bank, NeighborsDataset
from losses.contrastive_loss import SupConLoss
from utils import clustering_score, view_generator, save_results, clustering_accuracy_score, save_model
import heapq
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from collections import deque
from numpy.linalg import norm
from methods.ligd.asy_get_llm_batches_ds import get_query
import asyncio
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):  
        return int(obj)
    elif isinstance(obj, np.floating): 
        return float(obj)
    elif isinstance(obj, np.ndarray):  
        return obj.tolist()
    elif isinstance(obj, dict):  
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):  
        return [convert_numpy_types(x) for x in obj]
    else:
        return obj  


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj



def nearest_neighbor_cal(feature_space):
    k = 3  # include self, two neighbors
    neighbors = NearestNeighbors(n_neighbors=k).fit(feature_space)
    distances, indices = neighbors.kneighbors(feature_space, return_distance=True)

    edges = []
    for i in range(len(indices)):
        for j in range(1, k):  
            neighbor_idx = indices[i][j]
            weight = distances[i][j]
            edges.append((i, neighbor_idx, weight))
    
    return edges


def centrality_ranking(subgraph):
    """
    c(v) = deg(v) - (1 / W(C)) * sum(w_vu)
    return dict :  {node: centrality_score} 。
    """
    centrality = {}


    total_weight = sum([data['weight'] for _, _, data in subgraph.edges(data=True)])
    if total_weight == 0:
        total_weight = 1e-8 

    for node in subgraph.nodes():
        degree = subgraph.degree(node)

        weight_sum = sum([data['weight'] for _, _, data in subgraph.edges(node, data=True)])
 
        centrality_score = degree - (weight_sum / total_weight)
        centrality[node] = centrality_score

    return centrality


def process_graph(args, sparse_graph , selected_unlabeled_ex_indices, num_labels):
    """
    certral sample ranking
    """

    connected_components = list(nx.connected_components(sparse_graph))
    print("connected_components")
    print(len(connected_components))
    all_centralities = {}
    node_to_component_id = {}


    for component_id, component in enumerate(connected_components):
        subgraph = sparse_graph.subgraph(component)
        subgraph_centrality = centrality_ranking(subgraph)

        for node, score in subgraph_centrality.items():
            all_centralities[node] = score
            node_to_component_id[node] = component_id


    sorted_nodes = sorted(all_centralities.items(), key=lambda x: x[1], reverse=True)


    central_sample_nodes = []
    central_sample_component_ids = []

    for node, score in sorted_nodes:
        if node in selected_unlabeled_ex_indices:
            central_sample_nodes.append(node)
            central_sample_component_ids.append(node_to_component_id[node])
        if args.dataset == 'stackoverflow':
            if len(central_sample_nodes) >= (150 *0.75):
                break

        else: 
            if len(central_sample_nodes) >= (num_labels *0.75):  # certral samples number

                break

    return central_sample_nodes, central_sample_component_ids

class ALManager:

    def __init__(self, args, data, model_path, logger_name='Discovery'):


        assert model_path is not None, 'Model is None'
        
        self.logger = logging.getLogger(logger_name)
        self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
        self.logger.info(self.device)

        self.num_labels = data.num_labels
        self.n_known_cls = data.n_known_cls
        self.logger.info('self.num_labels: %s', str(self.num_labels))
        self.logger.info('Number of known classes: %s', str(self.n_known_cls))

        self.model = Bert(args)
        self.model.to(self.device)
        self.tokenizer = self.model.tokenizer
        
        self.load_pretrained_model(model_path)
        self.centroids = None

        self.prepare_data(args, data)

        # loss func
        self.cl_loss_fct = SupConLoss(temperature=0.07, contrast_mode='all',
                                      base_temperature=0.07)
        # optimizer and scheduler
        num_train_steps = int(len(self.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args, args.lr, num_train_steps)

        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        # self.test(self, self.test_dataloader, self.model, self.num_labels)
        results = self.test(test_dataloader=self.test_dataloader, model=self.model, num_labels=self.num_labels)

        if args.save_results:
            self.logger.info("***** Save results *****")
            results['epoch'] = -1
            save_results(args, results)


    def get_optimizer(self, args, lr, num_steps):
        num_warmup_steps = int(args.warmup_proportion*num_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
        return optimizer, scheduler


    def load_pretrained_model(self, model_path):
        
        if isinstance(model_path, str):
            self.logger.info('Loading pretrained model from %s', model_path)
            model_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_dict['model_state_dict'])
        elif isinstance(model_path, Bert):
            self.logger.info('Loading pretrained model from a Bert object')
            self.model.load_state_dict(model_path.state_dict())
        else:
            raise ValueError('model_path should be a str or a Bert object')


    def get_neighbor_dataset(self, args, dataset, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(dataset, indices)
        self.train_neighbor_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    
    def get_neighbor_inds(self, args, dataset, dataloader):
        """get indices of neighbors"""
        memory_bank = MemoryBank(
            len(dataset), 
            args.embed_feat_dim, len(self.all_label_list), 0.1)
        fill_memory_bank(dataloader, self.model, memory_bank, self.device)
        indices = memory_bank.mine_nearest_neighbors(args.topk, calculate_accuracy=False, gpu_id=args.gpu_id)
        return indices


    def get_adjacency(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj = torch.zeros(inds.shape[0], inds.shape[0])
        for b1, n in enumerate(neighbors):
            adj[b1][b1] = 1
            for b2, j in enumerate(inds):
                if j in n:
                    adj[b1][b2] = 1 # if in neighbors
                if (targets[b1] == targets[b2]) and (targets[b1]>0) and (targets[b2]>0):
                    adj[b1][b2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj
    def save_semi_dataset(self, dataset, filepath):

        def default_converter(o):

            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            else:
                return str(o)  
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(
                    dataset,
                    f,
                    indent=4,  
                    ensure_ascii=False, 
                    default=default_converter 
                )
            print(f"save in {filepath}")
        except TypeError as e:
            print(f"TypeError : {str(e)}")
        except IOError as e:
            print(f"IOError: {str(e)}")

    def prepare_data(self, args, data):
        
        new_data = NIDData(args, data, self.tokenizer)

        self.all_label_list = new_data.all_label_list

        self.train_semi_dataset = new_data.train_semi_dataset
        # self.save_semi_dataset(new_data.train_semi_ex_list, "new_data_train_semi_ex_list.json")

        self.logger.info(self.train_semi_dataset[0])
        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )
        
  
        self.llm_labels = np.array([ex['label_id'] for ex in new_data.train_semi_dataset])
        self.labeled_ex_labels = np.array([ex['label_id'] for ex in new_data.train_labeled_dataset])

        unique_labels = np.unique(self.labeled_ex_labels)
        self.logger.info(f'known_unique_labels:{unique_labels}')

        self.train_labeled_dataset = new_data.train_labeled_dataset
        self.train_labeled_examples = new_data.train_labeled_examples
        self.train_labeled_ex_list = new_data.train_labeled_ex_list
        train_labeled_sampler = SequentialSampler(self.train_labeled_dataset)
        self.train_labeled_dataloader = DataLoader(
            self.train_labeled_dataset,
            sampler=train_labeled_sampler,
            batch_size=args.eval_batch_size
        )


        self.train_unlabeled_dataset = new_data.train_unlabeled_dataset
        self.train_unlabeled_examples = new_data.train_unlabeled_examples
        self.train_unlabeled_ex_list = new_data.train_unlabeled_ex_list
        train_unlabeled_sampler = SequentialSampler(self.train_unlabeled_dataset)
        self.train_unlabeled_dataloader = DataLoader(
            self.train_unlabeled_dataset,
            sampler=train_unlabeled_sampler,
            batch_size=args.eval_batch_size
        )


        self.test_dataset = new_data.test_dataset
        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size
        )

    def al_finetune(self, args):

        self.logger.info('Start active learning finetune ...')
        self.llm_labeling(args, epoch=0, model=self.model)


        best_model = copy.deepcopy(self.model)
        best_metrics = {
            'Epoch': 0,
            'ACC': 0,
            'ARI': 0,
            'NMI': 0
        }
        self.logger.info("Start train ...")
        indices = self.get_neighbor_inds(args, self.llm_augmented_dataset, self.llm_augmented_dataloader)
        self.get_neighbor_dataset(args, self.llm_augmented_dataset, indices)


        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(self.train_neighbor_dataloader, desc="Iteration"):
                # 1. load data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adjacency(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. obtaining different views
                anchor_input_ids = batch["anchor_input_ids"].to(self.device)
                anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
                neighbor_input_ids = batch["neighbor_input_ids"].to(self.device)
                neighbor_attention_mask = batch["neighbor_attention_mask"].to(self.device)

                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):

                    anchor_sent_embed = self.model(
                        input_ids=anchor_input_ids, 
                        attention_mask=anchor_attention_mask, 
                        mode='simple_forward'
                    )
                    neighbor_sent_embed = self.model(
                        input_ids=neighbor_input_ids, 
                        attention_mask=neighbor_attention_mask, 
                        mode='simple_forward'
                    )
                    
                    sent_embed = torch.stack([anchor_sent_embed, neighbor_sent_embed], dim=1)

                    loss = self.cl_loss_fct(sent_embed, mask=adjacency)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
                    tr_loss += loss.item()
                    nb_tr_examples += anchor_input_ids.size(0)
                    nb_tr_steps += 1

                    
            tr_loss = tr_loss / nb_tr_steps
            self.logger.info("***** Epoch: %s: Train loss: %f *****", str(epoch), tr_loss)
            results = self.test(test_dataloader=self.test_dataloader, model=self.model, num_labels=self.num_labels)

            if args.save_results:
                self.logger.info("***** Save results *****")
                results['epoch'] = epoch
                save_results(args, results)

            if results['ACC'] + results['ARI'] + results['NMI'] > \
                best_metrics['ACC'] + best_metrics['ARI'] + best_metrics['NMI']:
                best_metrics['Epoch'] = epoch
                best_metrics['ACC'] = results['ACC']
                best_metrics['ARI'] = results['ARI']
                best_metrics['NMI'] = results['NMI']
                best_model = copy.deepcopy(self.model)
                save_model(args, best_model, epoch)

            self.logger.info("***** Curr and Best model metrics *****")
            self.logger.info(
                "Curr Epoch %s Test Score: ACC = %s | ARI = %s | NMI = %s", 
                str(epoch), str(results['ACC']), str(results['ARI']), str(results['NMI'])
            )
            self.logger.info(
                "Best Epoch %s Test Score: ACC = %s | ARI = %s | NMI = %s", 
                str(best_metrics['Epoch']), str(best_metrics['ACC']), str(best_metrics['ARI']), str(best_metrics['NMI'])
            )

            # update neighbors every several epochs
            if ((epoch + 1) % args.update_per_epoch) == 0:
                self.llm_labeling(args, epoch=epoch, model=best_model)
                indices = self.get_neighbor_inds(args, self.llm_augmented_dataset, self.llm_augmented_dataloader)
                self.get_neighbor_dataset(args, self.llm_augmented_dataset, indices)
            
            # if epoch == int(args.num_train_epochs) - 1 :
            #     save_model(args, best_model, epoch)



    def alignment(self, old_centroids, new_centroids, cluster_labels):
        self.logger.info("***** Conducting Alignment *****")
        if old_centroids is not None:

            old_centroids = old_centroids
            new_centroids = new_centroids
            
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)
            
            aligned_centroids = np.zeros_like(old_centroids)
            alignment_labels = list(col_ind)

            for i in range(self.num_labels):
                label = alignment_labels[i]
                aligned_centroids[i] = new_centroids[label]
    
            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in cluster_labels])

        else:
            aligned_centroids = new_centroids    
            pseudo_labels = cluster_labels 

        self.logger.info("***** Update Pseudo Labels With Real Labels *****")
        
        return aligned_centroids, pseudo_labels


    def llm_labeling(self, args, epoch, model):
        k = 0
        if not os.path.exists(os.path.join(args.result_dir, f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json')):
            self.logger.info(f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json')
            os.makedirs(os.path.dirname(os.path.join(args.result_dir, f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json')), exist_ok=True)
            
            self.logger.info('Start LLM labeling ...')
      
            feats, y_true = self.get_outputs(dataloader=self.train_semi_dataloader, model=model)

            self.logger.info('self.num_labels: %s', str(self.num_labels))
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            cluster_centroids, y_pred = km.cluster_centers_, km.labels_
    
            # choose unlabeled data
            selected_unlabeled_ex_indices = np.where(self.llm_labels == -1)[0]


            llm_generated_outputs = {
                "utterance_ori_inds": [],
                "true_cluster_ids": [],
                "pred_cluster_ids": [],
                "llm_pred_cluster_ids": [],
                "unc_neighbors_ori_inds": [],
                "unc_neighbors_labs": [],
            }
            price_usage = 0
          
            edges = nearest_neighbor_cal(feats)
            # self.logger.info(f'edges:{edges}')
            sparse_graph = nx.Graph()
            sparse_graph.add_weighted_edges_from(edges)
            central_samples, central_sample_component_ids = process_graph(args, sparse_graph, selected_unlabeled_ex_indices, self.num_labels )


    
            uncertainties = self.uncertainty_cal(args, feats, y_pred)
            uncertain_indices = [idx for idx, _ in sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)]
            top_n_uncertain_indices = []

            for i in uncertain_indices :
                if i in selected_unlabeled_ex_indices:
                    top_n_uncertain_indices.append(i)

                if args.dataset == 'stackoverflow':
                    if len(top_n_uncertain_indices) >= (150 *0.25):
                        break

                else: 
                    if len(top_n_uncertain_indices) >= (self.num_labels *0.25):  # number of hard negatives

                        break

            
            all_valid_u = self.hard_prob(args, top_n_uncertain_indices, sparse_graph, uncertainties,feats,y_pred ,epoch,'u')

            all_valid_c = self.hard_prob(args, central_samples, sparse_graph, uncertainties,feats,y_pred ,epoch,'c')


            all_selected_samples_id = top_n_uncertain_indices + central_samples
            all_valid = all_valid_u + all_valid_c

            
            true_cluster_ids = y_true[all_selected_samples_id]


            neighbor_labels = [
                [ int(true_cluster_ids[i]) ] * len(all_valid[i])
                for i in range(len(list(true_cluster_ids)))
            ]

        
            
            selected_labeled_ex_indices = np.where(self.llm_labels != -1)[0]
            selected_labeled_feats = feats[selected_labeled_ex_indices]
            selected_labeled_id_llm = [self.train_semi_dataset[idx]['label_id'] for idx in selected_labeled_ex_indices]
        
            k = len(set(selected_labeled_id_llm))
            km = KMeans(n_clusters = k).fit(selected_labeled_feats)
            selected_labeled_cluster_centroids, selected_labeled_y_pred = km.cluster_centers_, km.labels_
            
            selected_labeled_nearest_centroid_ex_inds_in_selected = self.get_nearst_centroid_example_no_label(selected_labeled_feats, selected_labeled_cluster_centroids, num_examples=1)    
          
            selected_labeled_nearest_centroid_ex_inds = selected_labeled_ex_indices[selected_labeled_nearest_centroid_ex_inds_in_selected]
         
            # choose k/2 representative samples to prompt , k is the number of the type of known labels in this epoch 
            print("selected_labeled_id_llm",selected_labeled_id_llm)
            
            prompts_stage_1 = {}

            for idx in all_selected_samples_id:
                target_feat = feats[idx]
                centroid_feats = feats[selected_labeled_nearest_centroid_ex_inds]
                distances = np.linalg.norm(target_feat - centroid_feats, axis=1)
             
                nearest_indices = np.argsort(distances)[:int(k / 2)]
             
                k_input_text_to_nearest_centroid_label = {}

                for sub_idx in nearest_indices:
             
                    actual_idx = selected_labeled_nearest_centroid_ex_inds[sub_idx]
                    input_text = self.train_semi_dataset[actual_idx]['input_text']
                    nearest_centroid_label = self.train_semi_dataset[actual_idx]['label_id']
      
                    k_input_text_to_nearest_centroid_label[input_text] = nearest_centroid_label

                prompt_string = "Below are representative samples from different classes:\n"
                prompt_string += "\n".join(f"{value}:{key}" for key, value in k_input_text_to_nearest_centroid_label.items())
                prompt_string += f"""These samples cover a variety of semantic themes. Your task is to select the two samples from the list above that are most semantically similar to the input sample based on their content.
                                If none of the samples are sufficiently semantically similar to the input sample, return -1.
                                Input Sample: {self.train_semi_dataset[idx]['input_text']}
                                Please provide the class numbers of the two selected samples, responding in the following format:\n
                                "Class X: sentence 1\n Class  Y: sentence 2"
                                If no sufficiently similar samples are found, respond with:
                                "None"
                                """
                prompts_stage_1[idx] = prompt_string
            result_stage_1 = asyncio.run(get_query(prompts_stage_1))
            prompts_stage_2 = {}
            new_label_results = {}

            for idx, classes in result_stage_1.items():
                if classes is None:
                    new_label_results[idx] = self.train_semi_dataset[key]['label_id_true']
                else:
                    prompts_stage_2[idx] = f"""Which one representative samples is more similar to the input sample?
                            Input Sample:  {self.train_semi_dataset[idx]['input_text']}: 
                            Representative samples: {classes}
                            Else please provide the class number of the similar sample in the following format:
                            "Class X"
                            """
            
            result_stage_2 = asyncio.run(get_query(prompts_stage_2))
            known_result_dict = {key: extract_integers(value)[0] if len(extract_integers(value)) > 0 else -1 for key, value in result_stage_2.items()}
            result_dict = {**known_result_dict,**new_label_results}
            result_ordered = [result_dict.get(idx) for idx in all_selected_samples_id]


            llm_generated_outputs["utterance_ori_inds"] = all_selected_samples_id
            llm_generated_outputs["true_cluster_ids"] = list(true_cluster_ids)
        
            llm_generated_outputs["llm_pred_cluster_ids"] = result_ordered
            
            llm_generated_outputs["unc_neighbors_ori_inds"] = all_valid
            llm_generated_outputs["unc_neighbors_labs"] = neighbor_labels

  
            split_idx = len(top_n_uncertain_indices)
            top_valid = all_valid[:split_idx]
            central_valid = all_valid[split_idx:]

            top_labels = true_cluster_ids[:split_idx]
            central_labels = true_cluster_ids[split_idx:]

            def compute_accuracy(valid_list, label_list):
                valid_flat = []
                preds = []
                for i in range(len(valid_list)):
                    valid_flat.extend(valid_list[i])
                    preds.extend([label_list[i]] * len(valid_list[i]))
                true_labels = y_true[valid_flat]
                correct = sum([1 for pred, true in zip(preds, true_labels) if pred == true])
                total = len(true_labels)
                return correct, total, correct / total if total > 0 else 0.0

     
            top_correct, top_total, top_acc = compute_accuracy(top_valid, top_labels)
            central_correct, central_total, central_acc = compute_accuracy(central_valid, central_labels)

       
            all_valid_flat = []
            all_preds = []
            for i in range(len(all_valid)):
                cluster_label = true_cluster_ids[i]
                neighbor_indices = all_valid[i]
                all_valid_flat.extend(neighbor_indices)
                all_preds.extend([cluster_label] * len(neighbor_indices))

            all_true = y_true[all_valid_flat]
            correct = sum([1 for pred, true in zip(all_preds, all_true) if pred == true])
            total = len(all_true)
            accuracy = correct / total if total > 0 else 0.0

      
            self.logger.info(f'Total samples: {total}')
            self.logger.info(f'Correct predictions: {correct}')
            self.logger.info(f'Accuracy: {accuracy:.4f}')

            self.logger.info(f'Top uncertain samples: {top_total}, Correct: {top_correct}, Accuracy: {top_acc:.4f}')
            self.logger.info(f'Central samples: {central_total}, Correct: {central_correct}, Accuracy: {central_acc:.4f}')
            llm_generated_outputs = convert_numpy_types(llm_generated_outputs)
     
            with open(os.path.join(args.result_dir, f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'), "w") as fp:
                json.dump(llm_generated_outputs, fp, indent=4,  default=to_serializable)

        else:
            self.logger.info('Loading LLM annotated output from %s', os.path.join(args.result_dir, f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'))
            with open(os.path.join(args.result_dir, f'1_ds_graph_llm_annotated_output_{args.seed}_{args.known_cls_ratio}_{epoch}.json'), "r") as fp:
                llm_generated_outputs = json.load(fp)
        # update label
        self.logger.info('updating labels')
        self.updating_dataset(args, llm_generated_outputs,k)

    def prob_knn(self, args, all_selected_samples_id, sparse_graph, cluster_centroids, feats,y_pred,epoch, flg):

        all_valid = []
        k = 100  
        neighbors = NearestNeighbors(n_neighbors=k).fit(feats)
        distances, indices = neighbors.kneighbors(feats, return_distance=True)
        for sample_id in all_selected_samples_id:
            valid = []
            neighbors = indices[sample_id]
            for neighbor in neighbors:
                if (neighbor != sample_id )and self.train_semi_dataset[neighbor]['label_id'] == -1 :
                    if len(valid) < 25:
                        valid.append(neighbor)
            all_valid.append(valid)

        return all_valid



    def hard_prob(self, args, all_selected_samples_id, sparse_graph, uncertainties, feats,y_pred,epoch, flg):
        # hard_prob
        all_valid = []  
        for sample_id in all_selected_samples_id:
            valid = []
            visited = set()
            queue = deque()
            
            # 初始节点的Uncertainty
            current_uncertainty = uncertainties[sample_id]
            
            # 处理初始节点的直接邻居
            neighbors = list(sparse_graph.neighbors(sample_id))
            for neighbor in neighbors:
                if self.train_semi_dataset[neighbor]['label_id'] == -1 :
                # and uncertainties[neighbor] >= current_uncertainty):
                    if neighbor not in visited:
                        valid.append(neighbor)
                        visited.add(neighbor)
                        queue.append((neighbor, uncertainties[neighbor]))

            if not valid:
                valid.append(sample_id)

            all_valid.append(valid)
            self.logger.info('-------------------------')
        
        return all_valid
    
    def hard_prob_f(self, args, all_selected_samples_id, sparse_graph, uncertainties, feats,y_pred,epoch, flg):
        # hard_prob
        all_valid = []  
        for sample_id in all_selected_samples_id:
            valid = []
            visited = set()
            queue = deque()
            
            # 初始节点的Uncertainty
            current_uncertainty = uncertainties[sample_id]
            
            # 处理初始节点的直接邻居
            neighbors = list(sparse_graph.neighbors(sample_id))
            for neighbor in neighbors:
                if self.train_semi_dataset[neighbor]['label_id'] == -1 :
                # and uncertainties[neighbor] >= current_uncertainty):
                    if neighbor not in visited:
                        valid.append(neighbor)
                        visited.add(neighbor)
                        queue.append((neighbor, uncertainties[neighbor]))

            if not valid:
                valid.append(sample_id)

            cur_feat = feats[sample_id]
            cur_valid_feats = [feats[i] for i in valid]

            cur_valid_feats = np.array(cur_valid_feats)  # shape: (N, 768)
            cur_feat = np.array(cur_feat)               # shape: (768,)
            # self.logger.info(f"cur_feat shape:{cur_feat.shape},cur_valid_feats shape: {cur_valid_feats.shape}")

            # if len(set(cur_valid_lael)) != 1:
            a_norm = cur_feat /norm(cur_feat)
            b_norm = cur_valid_feats / norm(cur_valid_feats, axis=1, keepdims=True)  # shape: (N, 768)

            cos_sim = np.dot(b_norm, a_norm)


            keep = filter_neighbors_unsupervised(cos_sim, method="mean", k=None, alpha=1)
            top_valid = [valid[i] for i in keep]


            all_valid.append(top_valid)

            self.logger.info('-------------------------')
        
        return all_valid

  
    def uncertainty_oneNode(self, predict_labels, k_nearest_neighbor,k):
        import math
        dict={}
        for i in range(len(k_nearest_neighbor)):
            point=k_nearest_neighbor[i]
            if predict_labels[point] not in dict.keys():
                dict[predict_labels[point]]=[point]
            else:
                dict[predict_labels[point]].append(point)
        sum=0
        for m in dict.keys():
            proportion=len(dict[m])/k
            if proportion != 0:
                sum = sum + proportion * math.log2(proportion)
        sum = -sum
        if sum==-0.0:
            sum=0.0
        return sum

    def uncertainty_cal(self,args, data, predict_labels):
        k = args.topk
        neighbors = NearestNeighbors(n_neighbors=k).fit(data)
        k_nearest_neighbors = neighbors.kneighbors(data, return_distance=False)
        uncertainty_dict=dict()
        for candidate in range(len(data)):
            k_nearest_neighbor=k_nearest_neighbors[candidate]
            uncertainty=self.uncertainty_oneNode(predict_labels, k_nearest_neighbor,args.topk)
            uncertainty_dict[candidate]=uncertainty
        self.logger.info(f'len(data):{len(data)}')
        self.logger.info(f'len(uncertainty_dict):{len(uncertainty_dict)}')
        return uncertainty_dict


    
    def updating_dataset(self, args, llm_generated_outputs,k):

        utterance_ori_inds = llm_generated_outputs["utterance_ori_inds"]
        true_cluster_ids = llm_generated_outputs["true_cluster_ids"]
        # pred_cluster_ids = llm_generated_outputs["pred_cluster_ids"]
        llm_pred_cluster_ids = llm_generated_outputs["llm_pred_cluster_ids"]
        unc_neighbors_ori_inds = llm_generated_outputs["unc_neighbors_ori_inds"]
        unc_neighbors_labs = llm_generated_outputs["unc_neighbors_labs"]

        self.logger.info('Num of labeled examples before LLM predicted: %s', str(len(np.where(self.llm_labels != -1)[0])))
        for i, u_ori_ind in enumerate(utterance_ori_inds):
            # double check for correctness
            assert self.train_semi_dataset[u_ori_ind]['label_id_true'] == true_cluster_ids[i]
            # update self.llm_labels
            llm_pred_cluster_id = llm_pred_cluster_ids[i]
            self.llm_labels[u_ori_ind] = llm_pred_cluster_id
            # assign labels to neighbors
            if len(unc_neighbors_ori_inds[i]) >0:
                for j, neighbor_ori_ind in enumerate(unc_neighbors_ori_inds[i]):
                    assert len(unc_neighbors_ori_inds[i]) == len(unc_neighbors_labs[i])
                    self.llm_labels[neighbor_ori_ind] = unc_neighbors_labs[i][j]
        
        self.logger.info('Num of labels after LLM predicted: %s', str(k))
        self.logger.info('Num of labeled examples after LLM predicted: %s', str(len(np.where(self.llm_labels != -1)[0])))

        # update dataset
        llm_augmented_list = []
        for ind, llm_label in enumerate(self.llm_labels):
            self.train_semi_dataset[ind]['label_id'] = llm_label

            if llm_label != -1:
                llm_augmented_list.append(self.train_semi_dataset[ind])

        train_semi_sampler = SequentialSampler(self.train_semi_dataset)
        self.train_semi_dataloader = DataLoader(
            self.train_semi_dataset,
            sampler=train_semi_sampler,
            batch_size=args.train_batch_size
        )

        self.llm_augmented_dataset = NIDDataset(llm_augmented_list)
        llm_augmented_sampler = SequentialSampler(self.llm_augmented_dataset)
        self.llm_augmented_dataloader = DataLoader(
            self.llm_augmented_dataset,
            sampler=llm_augmented_sampler,
            batch_size=args.train_batch_size
        )

    
    def get_nearst_centroid_example(self, feats, cluster_centroids, cluster_map, num_examples=1):
        self.logger.info('Get utterances nearst to centroids')
        assert feats.shape[1] == cluster_centroids.shape[1]
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
        D, I = index.search(cluster_centroids, num_examples)
        I = I.flatten() # (num_centroids, num_examples) -> (num_centroids * num_examples)
        ex_labels = []
        for i, indice in enumerate(I):
            if indice < len(self.labeled_ex_labels):
                ex_labels.append(self.labeled_ex_labels[indice])
            else:
                ex_labels.append(cluster_map[i])
        return I, np.asarray(ex_labels)
    
    def get_nearst_centroid_example_no_label(self, feats, cluster_centroids, num_examples=1):
        self.logger.info('Get utterances nearst to centroids')
        assert feats.shape[1] == cluster_centroids.shape[1]
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
        D, I = index.search(cluster_centroids, num_examples)
        I = I.flatten() # (num_centroids, num_examples) -> (num_centroids * num_examples)
        return I


    def get_uncertainty(self, args, feats, cluster_centroids):
        '''https://github.com/THU-BPM/SelfORE/blob/master/adaptive_clustering.py'''
        self.logger.info('Calculating student t distribution')
        assert feats.shape[1] == cluster_centroids.shape[1]
        # get student t distribution
        distances = euclidean_distances(feats, cluster_centroids)**2 
   
        st_distribution = (1.0 + distances / args.student_t_freedom) ** (- (args.student_t_freedom + 1) / 2)
        st_distribution = st_distribution / np.sum(st_distribution, axis=1, keepdims=True)
        st_distribution = np.clip(st_distribution, 1e-12, 1.0) 
        # get entropies
        entropies = - np.sum(st_distribution * np.log(st_distribution), axis=1)
        # refine uncertainty with neighbors
        self.logger.info('Refining uncertainty with neighbors')
        index = faiss.IndexFlatL2(feats.shape[1])
        index.add(feats)
  
        D, I = index.search(feats, args.uncertainty_neighbour_num + 1)
        D, I = D[:, 1:], I[:, 1:] 
        similarity_scores = np.exp(-D*args.rho)
        weighted_similarity_scores = np.mean(entropies[I] * similarity_scores, axis=-1)
        entropies = entropies + weighted_similarity_scores
        return entropies, I, st_distribution, similarity_scores


    def get_hungray_aligment(self, y_pred, y_true):
        num_test_samples = len(y_pred)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D))
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        y_pred_map = []
        cluster_map = [0]*len(ind)
        cluster_map_opp = [0]*len(ind)
        for i in range(num_test_samples):
            yp = y_pred[i]
            y_pred_map.append(ind[yp][1])
        y_pred_map = np.asarray(y_pred_map)

        for item in ind:
            cluster_map[item[0]] = item[1]
            cluster_map_opp[item[1]] = item[0]
        cluster_map = np.asarray(cluster_map)
        cluster_map_opp = np.asarray(cluster_map_opp)
        assert np.all(cluster_map[cluster_map_opp] == np.arange(len(ind)))
        return y_pred_map, cluster_map, cluster_map_opp

    def test(self, test_dataloader, model, num_labels):
        
        feats, y_true = self.get_outputs(dataloader=test_dataloader, model=model)
        km = KMeans(n_clusters=num_labels).fit(feats)
        y_pred = km.labels_
    
        test_results = clustering_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        return test_results
    
    def get_outputs(self, dataloader, model):

        model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, model.config.hidden_size)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration", leave=False):
            batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.set_grad_enabled(False):
                sent_embed = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], 
                    labels=None, mode='feature_ext')
                total_labels = torch.cat((total_labels, batch['label_id_true']))
                total_features = torch.cat((total_features, sent_embed))
                
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        return feats, y_true


def filter_neighbors_unsupervised(cos_sims, method="mean", k=None, alpha=0.9):

    cos_sims = np.array(cos_sims)

    if method == "mean":
        threshold = alpha * cos_sims.mean()
        keep_idx = np.where(cos_sims >= threshold)[0]

    elif method == "median":
        threshold = alpha * np.median(cos_sims)
        keep_idx = np.where(cos_sims >= threshold)[0]

    elif method == "iqr":
        q1 = np.percentile(cos_sims, 25)
        q3 = np.percentile(cos_sims, 75)
        iqr = q3 - q1
        threshold = q1 + 1.5 * iqr  
        keep_idx = np.where(cos_sims >= threshold)[0]

    elif method == "topk":
        if k is None:
            raise ValueError("Must specify k for topk method")
        keep_idx = np.argsort(cos_sims)[-k:]

    else:
        raise ValueError(f"Unknown method {method}")

    return keep_idx.tolist()

def extract_integers(s):
  
    integers = re.findall(r"[-+]?\d+", s)
    return integers