import torch
import torch.nn.functional as F
import cv2 as cv
import os
import clip
import matplotlib.pyplot as plt
import re
import faiss
import numpy as np
#######phần này có thể bỏ để dùng hàm của thanhnhan#####

#sửa 2 cái embed theo model  của thanhnhan

def embed_text(text: str, model, device):      
    token = clip.tokenize(text).to(device)          
    with torch.no_grad():
        text_embeded = model.encode_text(token)
    return text_embeded 

#-------------------------------------------------  -----------------------------------------------------------
#faiss IP
def retrieve_top_k_matches(query, k, feature_list, model, device, index):
    # Load the pre-saved feature vectors

    # # Prepare the embeddings and corresponding image names
    # img_names = []
    # embeddings = []
    # for img_name, img_embedding in feature_list:
    #     img_names.append(img_name)
    #     normalized_embedding = img_embedding.to(device).numpy() / np.linalg.norm(img_embedding.to(device).numpy())
    #     embeddings.append(normalized_embedding)

    # # Convert embeddings list to a numpy array
    # embeddings_np = np.vstack(embeddings)

    # # Create FAISS index for inner product similarity
    # dimension = embeddings_np.shape[1]
    # index = faiss.IndexFlatIP(dimension)  # Inner Product similarity
    # index.add(embeddings_np)  # Add all embeddings to the index

    # Embed the query text and normalize
    query_embedding = embed_text(query,model,device).to(device).numpy()
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Search for the top K nearest neighbors
    distances, indices = index.search(query_embedding.reshape(1, -1), k)

    # Retrieve the top K matches (img_names corresponding to indices)
    top_k_matches = [(feature_list[i][0], distances[0][j]) for j, i in enumerate(indices[0])]

    return top_k_matches

import copy
#tách tên ảnh
def extract_prefix_suffix(filename):
    # Remove the file extension
    name_without_extension = filename.rsplit('.', 1)[0]

    # Regular expression to match the prefix and suffix
    match = re.match(r"^(.*)_(\d+)$", name_without_extension)
    #print("=============")
    #print(name_without_extension)
    if match:
        prefix = "_".join(match.group(1).split('_')[:-1])  # This will match the "keyframe-0_L01_V01" part
        suffix = int(match.group(2))  # This will match the "229" part and convert it to an integer
        #print(prefix,suffix)
        return prefix, suffix
    else:
        return None, None


def extract_second_last_segment(filename):
    # Split the filename by underscores
    parts = filename.split('_')

    # Combine the second-to-last segment with the file extension
    second_last_segment_with_extension = parts[-2]  + '_' + parts[-1].split('.')[0]      #"keyframe-0_L01_V01_001_229.jpg" returns 001.jpg
    
    return second_last_segment_with_extension


#res_img là ảnh đã lựa chọn ở thao tác trước 
def find_img(img_list, next_query, res_img:list, threshold:int, res:list):         
    if(len(res_img) == len(img_list)):
        res.append(list(res_img))                   #using deep copy to avoid shared pointer                                    
        return

    for i in range(len(img_list[next_query])):
        vid_name_2, frame_index_2 = extract_prefix_suffix(img_list[next_query][i][0])
        vid_name_1, frame_index_1 = extract_prefix_suffix(res_img[-1][0])

        # if(i!=0):
        #     vid_name_prev, frame_index_prev = extract_prefix_suffix(img_list[next_query][i-1][0])
        #     if vid_name_prev == vid_name_2 and abs(frame_index_2 - frame_index_prev) <= -1:                  #threshold cho việc coi 2 cảnh là giống nhau
        #         continue                                                                                        #phụ thuộc vào việc nhảy frame lúc đọc frame chứ ko phụ thuộc threshold
        #print(frame_index_2)
        if(frame_index_2 - frame_index_1 <= threshold and frame_index_2>=frame_index_1 and vid_name_2==vid_name_1):
            res_img.append(img_list[next_query][i])
            find_img(img_list, next_query+1,res_img,threshold,res)
            res_img.pop()
        
                
#do getKbest đã sort theo độ chính xác nên kết quả hàm này hàng trên sẽ gần đúng nhất
def retreive_N_query(query_list, feature_list, K:int, threshold:int, model, device, index):        #K: số ảnh trả về cho mỗi query
    img_list = []                                                                   #mỗi hàng là list các tuple (name, similarity) ứng với mỗi query, sort theo cosine giảm dần
    #number of best match pics
    N = 300
    for query in query_list:
        top_k_matches = retrieve_top_k_matches(query, N, feature_list, model, device, index)
        img_list.append(top_k_matches)
    
    res_img= []                                     #mảng lưu lựa chọn ảnh cũ (name, similarity)
    res = []
    img_list
    for i in range(len(img_list[0])):   
        res_img.append(img_list[0][i])   
        find_img(img_list,1,res_img,threshold,res)
        res_img.pop()
    # getting final result
    # img_name_list contain len(queryList) names
    updated_res = []
    print(len(res))
    for img_set in res:
        vid_name, start_frame = extract_prefix_suffix(img_set[0][0])
        _, end_frame = extract_prefix_suffix(img_set[-1][0])
        image_names = [extract_second_last_segment(img_name[0]) for img_name in img_set]
        # Calculate the average similarity
        avg_similarity = sum(img_name[1] for img_name in img_set) / len(img_set)
        updated_res.append((image_names, vid_name, start_frame, end_frame, avg_similarity))         #list of tuple (list of name, vid, start frame, end frame, avg_cosine)
    updated_res = sorted(updated_res, key=lambda x: x[4], reverse=False)[:K]
    # with open('out.txt', 'w') as f:
    #     # Iterate over each entry in updated_res
    #     for entry in updated_res:
    #         list_of_names, video_name, start_frame, end_frame, score = entry
            
    #         # Convert list_of_names to a comma-separated string
    #         names_str = ', '.join(list_of_names)
            
    #         # Write all data on a single line, separated by a delimiter (e.g., a tab or comma)
    #         f.write(f"Names: {names_str}, Video Name: {video_name}, Start Frame: {start_frame}, End Frame: {end_frame}, Score: {score}\n")
    #lọc các đoạn chồng nhau
    eta = 5                    #nếu abs(frame_start[i] - frame_start[j] <=eta thì chỉ chọn 1 cái)
    # Convert the list of tuples to a list of lists for mutability
    updated_res = [list(item) for item in updated_res]
    i = 0
    while i < len(updated_res):
        j = i + 1
        while j < len(updated_res):
            check = False                               #có xóa hàng j ko 
            if updated_res[j][1] == updated_res[i][1]:  # Same video
                if updated_res[j][2] <= updated_res[i][2] and updated_res[i][2] - updated_res[j][2] <= eta:
                    updated_res[i][0][0] = copy.deepcopy(updated_res[j][0][0])  # Update the first image name
                    updated_res[i][2] = updated_res[j][2]
                    check = True
                if updated_res[j][3] >= updated_res[i][3] and updated_res[j][3] - updated_res[i][3] <= eta:
                    updated_res[i][0][-1] = copy.deepcopy(updated_res[j][0][-1])  # Update the last image name
                    updated_res[i][3] = updated_res[j][3]
                    check = True
            #check = False
            if check == True :              #0: ko xóa, 1: có thể lầ xóa
                del updated_res[j]
            else:
                j += 1
        i += 1

    # Convert the list of lists back to a list of tuples if needed
    updated_res = [tuple(item) for item in updated_res]
    print(updated_res[0], threshold)
    return updated_res #[tuple([img_names], vid, start, end, avg_cosine)]
