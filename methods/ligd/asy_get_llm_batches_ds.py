import asyncio
from openai import AsyncOpenAI
import time
import aiohttp


BATCH_SIZE = 30  

API_KEY = "your_api_key"
base_url = "your_url"

async def async_process_batch(session, futures, batch, api_key, idx, query, max_retries=3):
    retries = 0

    while retries < max_retries:
        async with session.post(
            f"{base_url}/chat/completions",  
            json={
                "model": "deepseek-v3-250324", 
                "messages": [
                    {"role": "system", "content": "You are an expert in the banking domain with a clear and precise understanding of user intent classification in this field. Please help with user intent classification."},
                    {"role": "user", "content": query}
                ]
            },
            headers={"Authorization": f"Bearer {api_key}"}  
        ) as response:
            if response.status == 200:
                completion = await response.json()
                new_response = completion["choices"][0]["message"]["content"]
                futures[idx] = new_response
                # print(futures)
                return 
            else:
                retries += 1
                # print(f"Request failed with status {response.status}, retrying ({retries}/{max_retries})...")
                await asyncio.sleep(1)  
   
    futures[idx] = "None"

    print(f"Request failed after {max_retries} retries. Setting futures[{idx}] to 'None'.")
    # raise Exception(f"API request failed after {max_retries} retries with status {response.status}")


async def async_process_queries(queries):
    all_results = {}
    num_batches = (len(queries) + BATCH_SIZE - 1) // BATCH_SIZE 
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_batches):
            batch = {k: queries[k] for k in list(queries.keys())[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]}  
            futures = {}  
            api_key = API_KEY
            tasks = [async_process_batch(session,futures,batch, api_key, idx, query) for idx,query in batch.items()]
            await asyncio.gather(*tasks)
            while len(futures) < len(batch):
                await asyncio.sleep(1)
            all_results.update(futures)
    print(all_results)
    return all_results



async def get_query(queries):
    
    start_time = time.time() 
    results = await async_process_queries(queries)
    end_time = time.time() 
    for idx,result in results.items():
        print(idx)
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return results

# query = {
#     "idx1": "What is the capital of France?"
# }

# asyncio.run(get_query(query))
