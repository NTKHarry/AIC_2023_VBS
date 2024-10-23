import streamlit as st
import numpy as np
import torch
import clip
import os
from PIL import Image
import pickle
import faiss
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from translates1 import *
from sequence_retrieval import *
from map_key_frame import *
import json
import csv

#---------------------------------------------------------------------------------------#
# Commented sections indicate features currently under development.                     #
#                                                                                       #
# To run the file, use the command 'streamlit run src/Ui_news.py' in the terminal.      #
# Note: The terminal must be in the directory path where 'src/Ui_news.py' is located.   #
# Example: ~/Desktop/gift/AIC_EITA                                                      #
# If the code is not working, please contact Thanhnhan for assistance (or ChatGPT).     #
#---------------------------------------------------------------------------------------#

# Please set 'path_image' to the path of the folder containing the images.


path_image = r"E:\AIC\Keyframe"
path_pretrained = r"D:\AIC_data\trained_data\pretrained_chou_3.pt"
path_data_2 = r"D:\AIC_data\trained_data\data2_1.pkl"
path_map_keyframes = r"D:\AIC_data\map_keyframes\map-keyframes"
path_metadata = r"D:\AIC_data\metadata\metadata"


# path_map_keyframes: là link map-keyframes nằm trong phần dataset
# path_metadata: là link path_metadata nằm trong phần dataset
# path_data_2: là link trong phần model

st.set_page_config(layout="wide")

if 'initialized' not in st.session_state:
    st.tmp_show = ""
    st.session_state.initialized = True

    st.session_state.device = "cpu"
    st.session_state.model, st.session_state.preprocess = clip.load("ViT-B/32", device=st.session_state.device)

    st.session_state.image_features_list_tmp = torch.load(path_pretrained)

    st.session_state.image_features_list = np.array([item[1] for item in st.session_state.image_features_list_tmp], dtype = 'float64').reshape(-1,512)
    st.session_state.image_features_list /= np.linalg.norm(st.session_state.image_features_list,axis=1, keepdims=True)
    st.session_state.image_features_small = st.session_state.image_features_list

    st.session_state.image_features_list_id = np.array([item[0] for item in st.session_state.image_features_list_tmp]).reshape(-1)
    st.session_state.image_features_small_id = st.session_state.image_features_list_id

    st.session_state.similarities = []
    st.session_state.similarities_small = st.session_state.similarities
    st.session_state.txt = ""
    st.session_state.text_sreach = ""

    # doc_embeddings = pickle.load(open(path_doc, 'rb'))
    # d = 384
    # st.session_state.index = faiss.IndexFlatL2(d)
    # st.session_state.index.add(doc_embeddings)
    # # Tạo mô hình Sentence Transformer
    # model_name = 'all-MiniLM-L6-v2'
    # st.session_state.model_sen = SentenceTransformer(model_name)
    # st.session_state.model_sen.to(st.session_state.device)
    # st.session_state.sentence_2 = pickle.load(open(path_sen2, "rb"))
    #st.session_state.translate = Translation()
    st.session_state.session_remind = "secondary"
    st.session_state.on_state = "En"
    st.session_state.times = 1
    
    st.session_state.name_video = np.unique(np.array(["".join(img.split("_")[1:]).split('.')[0] for img in st.session_state.image_features_list_id]))
    st.session_state.name_video_original = np.unique(np.array(["_".join(img.split("_")[1:]).split('.')[0] for img in st.session_state.image_features_list_id]))
    st.session_state.name_video_original.sort()
    st.session_state.name_video.sort()
    st.session_state.disabled_option = False

    st.session_state.list_choose = []
    st.session_state.list_choose_feature = []


    st.session_state.faisslist = faiss.IndexFlatL2(512)
    #print(np.array(st.session_state.image_features_small).shape)
    #faiss.normalize_L2(st.session_state.image_features_small)
    st.session_state.threshold = 1
    st.session_state.faisslist.add(np.array(st.session_state.image_features_small).reshape(-1,512))

    st.session_state.frame_tmp = 0


    
    tmp2 = pickle.load(open(path_data_2,'rb'))
    st.session_state.data2 = []
    for i in range(len(tmp2)-1):
       tmp = list(tmp2[i])
       tmp[3] = tmp[3].lower()
       st.session_state.data2.append((tmp[0], tmp[1], tmp[2], tmp[3]))

    st.session_state.list_image = []
    st.session_state.sub = []
    st.session_state.file_result = 0
    st.session_state.list_L = [True for _ in range(24)]

cols = st.columns([2.5,1])
with cols[0]:
    st.title("CLIP: dataset - Image news")
with cols[1]:
    st.caption("Layout:")
    cols1 = st.columns([2,1])
    with cols1[0]:
        option = st.selectbox("",
        ("Normal", "Video"),
        label_visibility="collapsed",disabled=st.session_state.disabled_option)
    with cols1[1]:
        if st.button("Reload", type = "primary",use_container_width = True):
            st.rerun()

tab = st.tabs(["Submit", "Video", "Setting"])
def int_str(k):
        if k<10:
            return "00"+str(k)
        if k<100:
            return "0" + str(k)
        if k<1000:
            return str(k)

def show_image_video(folder_path,video,value):
    img = Image.open(folder_path + '/' + video + '/' + int_str(value)+'.jpg')
    st.image(img, use_column_width=True)
@st.experimental_dialog("Video",width = "large")
def video1(folder_path, video_path):
    st.title("_".join(video_path.split("_")[1:]))
    st.title("Fps: " + str(pd.read_csv(path_map_keyframes+"/"+ video_path.split("_")[1] + "_" + video_path.split("_")[2] + ".csv").iloc[0,2]))
    video = video_path.split("_")[0] + "/" + video_path.split("_")[1] + "_" + video_path.split("_")[2]
    frame = int(video_path.split("_")[3].split('.')[0])-1
    #print(video)
    pts_time = extract_2(str(frame),path_map_keyframes+"/"+ video_path.split("_")[1] + "_" + video_path.split("_")[2] + ".csv")
    # values = st.slider("Select a range of values",frame-5, frame+5)
    # st.write("Values:", values)
    #st.image(folder_path + '/' + video + '/' + int_str(values)+'.jpg')
    # frame_rate = 1
    # for value in range(values-5, values+5):
    #     img = Image.open(folder_path + '/' + video + '/' + int_str(value)+'.jpg')
    #     st.image(img, use_column_width=True)
    #     time.sleep(1/frame_rate)
     
    # tab = st.tabs(["List image", "1 image"])
    # with tab[0]:
    #     for value in range(frame-5, frame+5):
    #         try:
    #             show_image_video(folder_path, video, value)
    #         except:
    #             continue
    # with tab[1]:
    #     cols2 = st.columns([2,1,2])
    #     with cols2[0]:
    #         if st.button("--", key= "-- 1",use_container_width = True):
    #             st.session_state.frame_tmp = int(st.session_state.frame_tmp) - 1 
    #     with cols2[1]:
    #         if st.button("center",key='||',use_container_width = True):
    #             st.session_state.frame_tmp = frame
    #     with cols2[2]:
    #         if st.button("++",key ="++ 1",use_container_width = True):
    #             st.session_state.frame_tmp = int(st.session_state.frame_tmp) + 1
    #     try:
    #         st.image(folder_path + '/' + video + '/' + int_str(int(st.session_state.frame_tmp)) +'.jpg')
    #     except:
    #         st.write("Hết video")
    # if st.button("Exit", key = "Exit-dialog"):
    #     st.rerun()
    #print(path_metadata+"/"+ video_path.split("_")[1] + "_" + video_path.split("_")[2]+'.json')
    with open(path_metadata+"/"+ video_path.split("_")[1] + "_" + video_path.split("_")[2]+'.json', encoding='utf-8') as file:
        link = json.load(file)['watch_url'].split('=')[1]
    st.write(f"""
    <style>
    .responsive-iframe {{
        position: relative;
        padding-bottom: 56.25%; /* Tỉ lệ 16:9 */
        height: 0;
        overflow: hidden;
        max-width: 100%;
        background: #000;
    }}
    .responsive-iframe iframe {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }}
    </style>
    <div class="responsive-iframe">
        <iframe src="https://www.youtube.com/embed/{link}?start={pts_time-1}&autoplay=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"https://www.youtube.com/embed/{link}?start={pts_time-1}&autoplay=1")
    if st.button("Exit", key = "Exit-dialog"):
        st.rerun()

def change_state():
    if st.session_state.on_state == "Vi":
        st.session_state.on_state = "En"
    else:
        st.session_state.on_state = "Vi"

def ui_text_input(times:int):
    txt = st.text_area("",label_visibility = 'collapsed', key = str(times))
    return txt
with st.sidebar:
    st.header("Ember in the Ashes")
    txt = []
    cols = st.columns(3)
    with cols[0]:
        if st.button("Add", key = "add",use_container_width = True):
            st.session_state.times += 1
    with cols[1]:
        if st.button("Delete", key = "delete",use_container_width = True):
            st.session_state.times -= 1
            st.session_state.times = max(1,st.session_state.times)
    with cols[2]:
        if st.button("Reset", key = "reset_text_area", type = "primary",use_container_width = True):
            st.session_state.times = 1
    for i in range(st.session_state.times):
        st.caption(f"Input {i+1}:")
        txt.append(ui_text_input(i))
    if st.session_state.times > 1:
        threshold = st.slider("threshold", 0, 30, 0)
    
    cols = st.columns(3)
    with cols[0]:
        button_enter = st.button("Enter", type="primary",use_container_width = True)
    if button_enter and (st.session_state.session_remind == 'primary' or len(st.session_state.sub)):
        st.warning('Chưa reset dataset', icon="⚠️")
    with cols[1]:
        btn_reset = st.button("Reset",key = "reset", type = st.session_state.session_remind,use_container_width = True)
        if btn_reset:
            st.session_state.similarities_small = []
            st.session_state.similarities = []
            st.session_state.sub = []
            st.session_state.list_image = []
            st.session_state.image_features_small = st.session_state.image_features_list
            st.session_state.session_remind = "secondary"
            st.rerun()
    with cols[2]:
        on = st.toggle(st.session_state.on_state,on_change = change_state,value=st.session_state.on_state=="Vi")
        #print(on)
    uploaded_files = st.file_uploader("Choose a Image file",type=["jpg", "png", "jpeg"])
    button_submit = st.button("Submit", type="secondary")
    if uploaded_files is not None:
        image = Image.open(uploaded_files)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.title("Approximately find text")
    text_find = st.text_input("here",label_visibility = 'visible')
    button_text_find = st.button("Enter")
def show():
    
    num_best = len(st.session_state.list_choose)
    cols = st.columns(4)
    for col,j in zip(cols,range(len(cols))):
        with col:
            for i in range(j,num_best,len(cols)):
                st.image(os.path.join(path_image,st.session_state.list_choose[i][0].split("_")[0] + "/" + st.session_state.list_choose[i][0].split("_")[1] + "_" +st.session_state.list_choose[i][0].split("_")[2] + "/" + st.session_state.list_choose[i][0].split("_")[3]) + ".jpg")
                subcol = st.columns([1,2,1])
                with subcol[0]:
                    st.caption(f"Rank: {i+1}")
                with subcol[1]:
                    btn_video = st.button("_".join(st.session_state.list_choose[i][0].split("_")[1:]),use_container_width=True, key = "list_choose:" + st.session_state.list_choose[i][0])

                with subcol[2]:
                    btn_delete = st.button("Del", key = "D" + str(j) + " - " + str(i)+ "list_choose",use_container_width=True)
                if btn_delete:
                    #print(int(best_match_image_path[i,0].split('.')[0].split('_')[1]),int(best_match_image_path[i,0].split('.')[0].split('_')[2]))
                    #delete_image(int(list_image[i,0].split('.')[0].split('_')[1]),int(list_image[i,0].split('.')[0].split('_')[2]),3)
                    #print("Detele")
                    video, frame = st.session_state.list_choose[i][0].split('_')[1] + "_" + st.session_state.list_choose[i][0].split('_')[2], int(st.session_state.list_choose[i][0].split('_')[3].split('.')[0])

                    #frame = int(st.session_state.list_choose[i][0].split('.')[0].split('_')[2])
                    if len(st.session_state.similarities_small) != 0 and st.session_state.list_choose[i][0] not in np.array(st.session_state.similarities_small)[:,0]:
                        st.session_state.similarities_small.append(st.session_state.list_choose[i])
                        st.session_state.similarities_small.sort(key = lambda x:x[1],reverse=False)
                    st.session_state.list_choose =  [item for item in st.session_state.list_choose if  (item[0].split('_')[1] + "_" + item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1)]
                    st.rerun()
                if btn_video:
                    st.session_state.frame_tmp = st.session_state.list_choose[i][0].split('_')[3]
                    video1(path_image,st.session_state.list_choose[i][0])
                    

with tab[2]:
    st.session_state.num = st.slider("số lượng hình ảnh hiển thị", 100, 600, 200, step=4)
    cols = st.columns(2)
    with cols[0]:
        col1 = st.columns(2)
        with col1[1]:
            if st.button("Reset", type = "primary", use_container_width= True):
                st.session_state.list_L = [True for _ in range(24)]
        with col1[0]:
            st.caption("Choose L: (chờ đợi là hạnh phúc)")
        cols1 = st.columns(4)
        for i in range(1,25):
            with cols1[(i-1)%4]:
                st.session_state.list_L[i-1] = st.checkbox("L"+str(i),st.session_state.list_L[i-1])
  
with tab[0]:
    # st.session_state.disabled_option = True
    subcol = st.columns([1,1.5,1,0.5,1,1,2])
    with subcol[6]:
        btn_submit_result = st.button("Submit", key = "sumbit_result", type="primary", use_container_width=True)
        if btn_submit_result:
            res = [[x[0].split('_')[1] + "_" + x[0].split('_')[2], x[0].split('_')[4].split('.')[0]] for x in st.session_state.list_choose]
            st.session_state.file_result += 1

            with open("output" + '_' + str(st.session_state.file_result) +'.csv' , mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(res)
            st.session_state.list_choose = []
            #st.session_state.similarities = ""
            st.session_state.txt = ""
            st.session_state.text_sreach = ""
            #st.session_state.image_features_small = st.session_state.image_features_list
            st.session_state.button_likely = []
    with subcol[0]:
        st.subheader(f"Total: {len(st.session_state.list_choose)}")


    with subcol[5]:
        btn_auto_fill = st.button("Auto Fill", key = "auto-fill", type="secondary", use_container_width=True)
        if btn_auto_fill and len(st.session_state.similarities_small) != 0:
            lacksz = 100
            name = np.array([x[0] for x in st.session_state.similarities_small[:200]])
            _, candidate_index = np.unique(name,return_index=True)
            # print(candidate_index.shape)
            # print(candidate_index[:10])
            unique_list = unique_list = list(dict.fromkeys(map(tuple, np.array(st.session_state.list_choose + st.session_state.similarities_small))))
            st.session_state.list_choose = unique_list[:lacksz]
            st.session_state.similarities_small = unique_list[lacksz:]
            st.rerun()
    with subcol[4]:
        if st.button("Reset",key="reset-choose", use_container_width=True):
            st.session_state.list_choose = []
            if len(st.session_state.similarities_small) != 0:
                st.session_state.similarities_small += st.session_state.list_choose
                st.session_state.similarities_small.sort(key = lambda x:x[1],reverse=False)
            st.rerun()
    col_3 = st.columns([5,1,1,1,1])
    # with col_3[1]:
    #     st.caption("Insert")
    with col_3[1]:
        option_swap1 = st.selectbox("",
        range(1,len(st.session_state.list_choose)+1),
        index=None,
        label_visibility = "collapsed",
        placeholder="rank...", key = "option-insert-1")
    # with col_3[3]:
    #     st.caption("into")
    with col_3[2]:
        option_swap2 = st.selectbox("",
        range(1,len(st.session_state.list_choose)+1),
        index=None,
        label_visibility = "collapsed",
        placeholder="rank...",key = "option-insert-2")
    with col_3[3]:
        if st.button("insert", key = "insert-button",use_container_width=True):
            element_insert = st.session_state.list_choose.pop(int(option_swap1)-1)
            st.session_state.list_choose.insert(int(option_swap2)-1, element_insert)
    with col_3[4]:
        if st.button("swap", key = "swap-button",use_container_width=True):
            st.session_state.list_choose[int(option_swap1)-1], st.session_state.list_choose[int(option_swap2)-1] = st.session_state.list_choose[int(option_swap2)-1], st.session_state.list_choose[int(option_swap1)-1]

    def find_tmp(x, y):
            result = [tup[1] for tup in x if tup[0] == y]
            if len(result) == 0:
                return 100000000
            return result[0]

    with subcol[1]:
        option1 = st.text_input("Ex: L12_V001_100")
        if option1:
            print(str((int(option1.split('_')[-1])-1)*50))
            option1 = option1 + "_" + str((int(option1.split('_')[-1])-1)*50) +'.jpg'
    with subcol[2]:
        if st.button("5", use_container_width=True):
            for x in [0,1,-1,2,-2]:
                #print(option2)
                option2 = "_".join(option1.split('_')[:-2]) + "_" + int_str(int(option1.split('_')[-2]) + x) + "_" + str((int(option1.split('_')[-2]) + x - 1)*50)
                print(option2)
                distance = find_tmp(st.session_state.similarities_small, "keyframes-"+ str(int(option2.split('_')[0].split('L')[-1])-1) + "_" + option2)
                print(distance)
                if ("keyframes-"+ str(int(option2.split('_')[0].split('L')[-1])-1) + "_" + option2,distance) not in st.session_state.list_choose:
                    st.session_state.list_choose.append(("keyframes-"+ str(int(option2.split('_')[0].split('L')[-1])-1) + "_" + option2,distance))
                    print(("keyframes-"+ str(int(option2.split('_')[0].split('L')[-1])-1) + "_" + option2,distance))
                    video = option2.split('_')[1]
                    frame = option2.split('_')[2].split('.')[0]
                    video, frame = option2.split('_')[1], int(option2.split('_')[2].split('.')[0])
                    if len(st.session_state.similarities_small) != 0 and len(st.session_state.similarities_small[0][0]) != 2:
                        #print(st.session_state.similarities_small)
                        st.session_state.similarities_small = [item for item in st.session_state.similarities_small if (item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1)]
                    #st.session_state.image_features_small = [item for item in st.session_state.image_features_small if   (item[0].split('_')[1] != str(video) or abs(int(item[0].split('_')[2].split('.')[0]) - int(frame))  >= 1)]
                    #st.session_state.session_remind = "primary"
    with subcol[3]:
        btn_find_image = st.button("Add", use_container_width=True)
        if btn_find_image:
            #st.title(option1)
            # st.title(st.session_state.similarities_small[0])
            distance = find_tmp(st.session_state.similarities_small, "keyframes-"+ str(int(option1.split('_')[0].split('L')[-1])-1) + "_" + option1)
            if ("keyframes-"+ str(int(option1.split('_')[0].split('L')[-1])-1) + "_" + option1,distance) not in st.session_state.list_choose:
                
                st.session_state.list_choose.append(("keyframes-"+ str(int(option1.split('_')[0].split('L')[-1])-1) + "_" + option1,distance))
                video = option1.split('_')[1]
                frame = option1.split('_')[2].split('.')[0]
                video, frame = option1.split('_')[1], int(option1.split('_')[2].split('.')[0])
                if len(st.session_state.similarities_small[0][0]) != 2:
                    #print(st.session_state.similarities_small)
                    st.session_state.similarities_small = [item for item in st.session_state.similarities_small if (item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1)]
                #st.session_state.image_features_small = [item for item in st.session_state.image_features_small if   (item[0].split('_')[1] != str(video) or abs(int(item[0].split('_')[2].split('.')[0]) - int(frame))  >= 1)]
                #st.session_state.session_remind = "primary"
    show()


with tab[1]:
    def set_init():
        #st.session_state.similarities = ""
        st.session_state.txt = ""
        st.session_state.text_sreach = ""
        #st.session_state.image_features_small = st.session_state.image_features_list
        st.session_state.button_likely = []
        st.list_image = []
        st.sub = []
    if button_text_find:
        set_init()
        st.session_state.text_sreach = text_find
        if len(st.session_state.similarities):
            st.warning('Chưa reset dataset', icon="⚠️")

    if button_enter:
        #if st.session_state.session_remind != "primary":
        set_init()
        # print("check")
        if st.session_state.times >= 2:
            st.session_state.threshold = threshold
        st.session_state.txt = txt
    if button_submit:
        set_init()
        st.session_state.txt = ""

    def find_similarity(text_features):
        #faiss.normalize_L2(text_features)
        text_features /= np.linalg.norm(text_features,axis=1, keepdims=True)
        D, I = st.session_state.faisslist.search(text_features.reshape(1,-1),1000)
        similarities = st.session_state.image_features_small_id[I][0]
        similarities_tmp = list(zip(similarities,D[0]))
        st.session_state.similarities_small = similarities_tmp

        return similarities_tmp
    # find image - image 
    if uploaded_files is not None:
        image = Image.open(uploaded_files)
        image_input = st.session_state.preprocess(image).unsqueeze(0).to(st.session_state.device)
        with torch.no_grad():
            image_features = st.session_state.model.encode_image(image_input)
        st.session_state.similarities = find_similarity(image_features)
    # Text - image
    if st.session_state.txt != "":
        if on:
            text_translate = [generate(item) for item in st.session_state.txt]
        else:
            text_translate = st.session_state.txt
        st.tmp_show = text_translate
        if len(st.session_state.txt) == 1:
            #print("checkcu")
            st.session_state.disabled_option = False
            text_inputs = torch.cat([clip.tokenize(text_translate[0])]).to(st.session_state.device)
            with torch.no_grad():
                text_features = st.session_state.model.encode_text(text_inputs)
            st.session_state.similarities = find_similarity(text_features)
        else:
            #print("checkmoiw")
            st.session_state.disabled_option = True
            option = "Video"
 
            st.session_state.similarities = retreive_N_query(text_translate,st.session_state.image_features_list_tmp, K=200, threshold=st.session_state.threshold*25,model=st.session_state.model,device = st.session_state.device,index = st.session_state.faisslist)
            st.session_state.similarities_small = st.session_state.similarities
            if st.session_state.threshold == 0:
                tmp_simi = [(item[1] + "_" + item[0][0], item[-1]) for item in st.session_state.similarities]
                st.session_state.similarities = tmp_simi
                #print(tmp_simi[0][0])
                st.session_state.similarities_small = tmp_simi
        #print(f"st.session_state.disabled_option: {st.session_state.disabled_option}")
        st.session_state.txt = ""
        st.rerun()
    # find best

    st.sidebar.title(st.tmp_show)
    def delete_image(video, frame, threshold):
        st.session_state.similarities_small = [item for item in st.session_state.similarities_small if   (item[0].split('_')[1] + "_" + item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1)]
        #st.session_state.image_features_small = [item for item in st.session_state.image_features_small if   (item[0].split('_')[1] != str(video) or abs(int(item[0].split('_')[2].split('.')[0]) - int(frame))  >= 1)]
        st.session_state.session_remind = "primary"

    def change_image(video,frame):
        st.session_state.similarities_small = [item for item in st.session_state.similarities_small if   (item[0].split('_')[1] + "_" + item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1)]
        #st.session_state.image_features_small = [item for item in st.session_state.image_features_small if   (item[0].split('_')[1] != str(video) or abs(int(item[0].split('_')[2].split('.')[0]) - int(frame))  >= 1)]
        st.session_state.session_remind = "primary"

    
    def show_image():
        st.session_state.similarities_small = st.session_state.similarities
        if sum(st.session_state.list_L) <24:
            simi = []
            for i in range(len(st.session_state.similarities_small)):
                print(int(st.session_state.similarities_small[i][0].split('_')[1].split('L')[-1])-1)
                if st.session_state.list_L[int(st.session_state.similarities_small[i][0].split('_')[1].split('L')[-1])-1]:
                    simi.append(st.session_state.similarities_small[i])
                if len(simi) > st.session_state.num:
                    break
            st.session_state.similarities_small = simi

        num_best = min(st.session_state.num,len(st.session_state.similarities_small))
        #num_best = 200
        #print("check1")
        #print(len(st.session_state.button_likely))
        ans = st.session_state.similarities_small
        #ans = st.session_state.image_features_list_tmp
        best_match_image_path = ans[:num_best]
        cols = st.columns(4)
        for col,j in zip(cols,range(len(cols))):
            with col:
                for i in range(j,num_best,len(cols)):
                    #st.image(os.path.join(path_image,best_match_image_path[i][0]))
                    #print(best_match_image_path[i][0].split("_"))
                    st.caption(i)
                    st.image(path_image + "/" +best_match_image_path[i][0].split("_")[0] + "/" + best_match_image_path[i][0].split("_")[1] + "_" + best_match_image_path[i][0].split("_")[2] + "/" + best_match_image_path[i][0].split("_")[3] + ".jpg")

                    subcol = st.columns([1.5,1,1.5,1])
                    with subcol[0]:
                        btn_video = st.button("_".join(best_match_image_path[i][0].split("_")[1:-1]),use_container_width=True, key = best_match_image_path[i][0])
                    with subcol[1]:
                        btn_choose = st.button('++',use_container_width=True,key = "add"+best_match_image_path[i][0])
                    with subcol[2]:
                        btn_similarity = st.button("Find",key = str(j) + " - " + str(i),type = "primary",use_container_width = True)
                        #st.session_state.button_likely.append((os.path.join(path_image,best_match_image_path[i,0]),btn1))
                    with subcol[3]:
                        btn_delete = st.button("X", key = "D" + str(j) + " - " + str(i),use_container_width=True)

                    if btn_similarity:    
                        image = Image.open(path_image + "/" +best_match_image_path[i][0].split("_")[0] + "/" + best_match_image_path[i][0].split("_")[1] + "_" + best_match_image_path[i][0].split("_")[2] + "/" + best_match_image_path[i][0].split("_")[3] + ".jpg")
                        image_input = st.session_state.preprocess(image).unsqueeze(0).to(st.session_state.device)
                        with torch.no_grad():
                            image_features = st.session_state.model.encode_image(image_input)
                        set_init()
                        st.session_state.similarities_small = find_similarity(image_features)
                        #uploaded_files = os.path.join(path_image,best_match_image_path[i,0])
                        st.rerun()
                    if btn_delete:
                        #print(int(best_match_image_path[i,0].split('.')[0].split('_')[1]),int(best_match_image_path[i,0].split('.')[0].split('_')[2]))
                        delete_image(best_match_image_path[i][0].split('_')[1] + "_" + best_match_image_path[i][0].split('_')[2],int(best_match_image_path[i][0].split('_')[3].split('.')[0]),3)
                        #print("Detele")
                        st.rerun()
                    if btn_choose:
                        #print(st.session_state.similarities_small[i])
                        if st.session_state.list_choose == [] or st.session_state.similarities_small[i][0] not in np.array(st.session_state.list_choose)[:,0]:
                            st.session_state.list_choose.append(st.session_state.similarities_small[i])
                        change_image(best_match_image_path[i][0].split('_')[1] + "_" + best_match_image_path[i][0].split('_')[2],int(best_match_image_path[i][0].split('_')[3].split('.')[0]))
                        
                        st.rerun()
                    if btn_video:
                        st.session_state.frame_tmp = int(best_match_image_path[i][0].split('_')[3].split('.')[0])
                        video1(path_image,best_match_image_path[i][0])

    def classify_list(list_image):
        image = []
        for tmp in list_image:
            ok = False
            for class_item in image:
                check = False
                if len(class_item) >= 5:
                    continue
                for i in range(len(class_item)):
                    if (class_item[i][0].split('_')[1] + class_item[i][0].split('_')[2] == tmp[0].split('_')[1] + tmp[0].split('_')[2]) and abs(int(class_item[i][0].split('_')[3].split('.')[0])  - int(tmp[0].split('_')[3].split('.')[0])) <= 10:
                        check = True
                        break

                if not check:
                    continue
                class_item.append(tmp)
                ok = True
                break
            if not ok:
                image.append([tmp])
        return image

    def show_image_2():
        num_best = 200
        #print("check1")
        #print(len(st.session_state.button_likely))
        #ans = np.array(st.session_state.similarities_small)
        #print(ans)
        #best_match_image_path = st.session_state.similarities_small[:num_best]
        list_image = st.session_state.similarities_small[:num_best] 
        print(6 // len(list_image[0][0]))
        jj = 0
        for i in range(0,len(list_image),6 // len(list_image[0][0])):
            col_21 = st.columns(6 // len(list_image[0][0]))
            for ii in range(6// len(list_image[0][0])):
                with col_21[ii]:
                    #print(list_image[i+ii][0])
                    if jj < len(list_image):

                        st.write(list_image[i+ii][1])

                        cols2 = st.columns(len(list_image[i+ii][0]))
                        jj1 = 0
                        #print(list_image[i+ii])
                        for j in range(len(list_image[i+ii][0])):
                            #list_image[i].sort(key=lambda x: int(x[0].split('_')[2].split('.')[0]))
                            #print(list_image[i][3] - list_image[i][2]+1)
                            with cols2[jj1]:
                                st.image(path_image + "/" + list_image[i+ii][1].split("_")[0] + "/" + list_image[i+ii][1].split("_")[1] + "_" + list_image[i+ii][1].split("_")[2] + "/" + list_image[i+ii][0][j].split('_')[0] +".jpg")                    
                                st.caption(str(list_image[i+ii][0][j].split('_')[0]))
                                if st.button("++",key =str(i) + str(ii) + str(j)):
                                    
                                    st.session_state.list_choose.append((list_image[i+ii][1] + "_" + list_image[i+ii][0][j].split('_')[0] + "_" + str((int(list_image[i+ii][0][j].split('_')[0])-1) * 50) ,-1))
                                    st.rerun()
                            jj1+=1
                        jj += 1
                # with cols2[jj]:
                #     st.image(path_image + "/" + list_image[i][1].split("_")[0] + "/" + list_image[i][1].split("_")[1] + "/" + list_image[i][0][0])
                #     st.caption(str(list_image[i][3]))
        st.divider()
    def show_image_2_1_query():
        num_best = 200
        #print("check1")
        ans = np.array(st.session_state.similarities_small)
        best_match_image_path = ans[:num_best]
        list_image = classify_list(best_match_image_path)
        for i in range(len(list_image)):
            st.caption(list_image[i][0][0].split("_")[1] + "_" + list_image[i][0][0].split("_")[2])
            cols = st.columns(max(2,len(list_image[i])))
            list_image[i].sort(key=lambda x: int(x[0].split('_')[3].split('.')[0]))
            for j in range(min(len(list_image[i]),len(cols))):
                with cols[j]:
                    st.image(path_image + "/" +list_image[i][j][0].split("_")[0] + "/" + list_image[i][j][0].split("_")[1] + "_" + list_image[i][j][0].split("_")[2] + "/" + list_image[i][j][0].split("_")[3]+".jpg")
                    col2 = st.columns([1,3])
                    with col2[0]:
                        st.caption(list_image[i][j][0].split('_')[3].split('.')[0])
                    with col2[1]:
                        btn_choose = st.button('++',use_container_width=True,key = "add" + list_image[i][j][0] + "-2")
                    if btn_choose:
                        if st.session_state.list_choose == [] or list_image[i][j][0] not in np.array(st.session_state.list_choose)[:,0]:
                            st.session_state.list_choose.append(list_image[i][j])
                        #change_image(best_match_image_path[i][0].split('_')[1] + "_" + best_match_image_path[i][0].split('_')[2],int(best_match_image_path[i][0].split('_')[3].split('.')[0]))
                        list_image[i].pop(j)
                        st.rerun()
            st.divider()
    if len(st.session_state.similarities):
        st.session_state.text_sreach = ""
        if st.session_state.disabled_option==False:
            if option == "Normal":
                show_image()
            else:
                show_image_2_1_query()
        else:
            if st.session_state.threshold != 0:
                show_image_2()
            else:
                show_image()
    #show_image()

    def time_k(time):
        return round(float(time.split(':')[1])*60 + float(time.split(':')[2]))

    def find_image_range(video, start, end):
        path = path_map_keyframes + "/" + video +'.csv'
        return extract_3(time_k(start), time_k(end),path)

    def show_image_3(list_image,sub):
        num_best = min(200,len(sub))
        #num_best = 200
        #print("check1")
        #print(len(st.session_state.button_likely))
        ans = list_image
        #ans = st.session_state.image_features_list_tmp
        best_match_image_path = ans[:num_best]
        cols = st.columns(4)
        #print(len(sub),len(list_image))
        for col,j in zip(cols,range(len(cols))):
            with col:
                for i in range(j,num_best,len(cols)):
                    #st.image(os.path.join(path_image,best_match_image_path[i][0]))
                    #print(best_match_image_path[i][0].split("_"))
                    st.caption(sub[i])
                    st.image(path_image + "/" +best_match_image_path[i][0].split("_")[0] + "/" + best_match_image_path[i][0].split("_")[1] + "_" + best_match_image_path[i][0].split("_")[2] + "/" + best_match_image_path[i][0].split("_")[3] + ".jpg")

                    subcol = st.columns([1.5,1])
                    with subcol[0]:
                        btn_video = st.button("_".join(best_match_image_path[i][0].split("_")[1:-1]),use_container_width=True, key = best_match_image_path[i][0])
                    with subcol[1]:
                        btn_choose = st.button('++',use_container_width=True,key = "add"+best_match_image_path[i][0])
                    # with subcol[2]:
                    #     btn_similarity = st.button("Find",key = str(j) + " - " + str(i),type = "primary",use_container_width = True)
                    #     #st.session_state.button_likely.append((os.path.join(path_image,best_match_image_path[i,0]),btn1))
                    # with subcol[3]:
                    #     btn_delete = st.button("X", key = "D" + str(j) + " - " + str(i),use_container_width=True)

                    if btn_choose:
                        #print(st.session_state.similarities_small[i])

                        if st.session_state.list_choose == [] or list_image[i][0] not in np.array(st.session_state.list_choose)[:,0]:
                            st.session_state.list_choose.append(list_image[i])
                        #change_image(best_match_image_path[i][0].split('_')[1] + "_" + best_match_image_path[i][0].split('_')[2],int(best_match_image_path[i][0].split('_')[3].split('.')[0]))
                        video,frame = best_match_image_path[i][0].split('_')[1] + "_" + best_match_image_path[i][0].split('_')[2],int(best_match_image_path[i][0].split('_')[3].split('.')[0])
                        tmp = []
                        sub_tmp = []
                        for i in range(len(list_image)):
                            item = list_image[i]
                            if (item[0].split('_')[1] + "_" + item[0].split('_')[2] != str(video) or abs(int(item[0].split('_')[3].split('.')[0]) - int(frame))  >= 1):
                                tmp.append(item)
                                sub_tmp.append(sub[i])
                        st.session_state.list_image = tmp
                        st.session_state.sub = sub_tmp
                        #st.session_state.image_features_small = [item for item in st.session_state.image_features_small if   (item[0].split('_')[1] != str(video) or abs(int(item[0].split('_')[2].split('.')[0]) - int(frame))  >= 1)]
                        # st.session_state.session_remind = "primary"
                        st.rerun()
                    if btn_video:
                        st.session_state.frame_tmp = int(best_match_image_path[i][0].split('_')[3].split('.')[0])
                        video1(path_image,best_match_image_path[i][0])
    #@st.cache_data
    def find_sub():
        st.session_state.similarities_small = []
        #set_init()
        st.session_state.list_image = [] 
        st.session_state.sub = []
        st.session_state.text_sreach = st.session_state.text_sreach.lower()
        for src_tmp in st.session_state.data2:
            if st.session_state.text_sreach in src_tmp[3]:

                video = src_tmp[0]
                start = src_tmp[1]
                end = src_tmp[2]
                list_image_ = find_image_range(video,start,end)
                #print(len(st.session_state.list_image))
                for img in list_image_:
                    # if ( "keyframes-"+str(int(video.split('_')[0].split('L')[1]) -1)+ "_" + video + "_" + int_str(img[0])+ "_" + str(img[1]) +'.jpg',10) in st.session_state.list_image:
                    #     con
                    st.session_state.list_image.append(( "keyframes-"+str(int(video.split('_')[0].split('L')[1]) -1)+ "_" + video + "_" + int_str(img[0])+ "_" + str(img[1]) +'.jpg',10))
                    st.session_state.sub.append(src_tmp[3])
                    if len(st.session_state.list_image) >= 200:
                        break
        #print(len(st.session_state.list_image))
        arr = np.array([x[0] for x in st.session_state.list_image])
        #print(arr[:4])
        unique_values, indices = np.unique(arr, return_index=True)
        #print(indices)
        st.session_state.list_image = np.array(st.session_state.list_image)[indices].tolist()

        st.session_state.sub = np.array(st.session_state.sub)[indices].tolist()
        st.session_state.text_sreach = ""
    if st.session_state.text_sreach != "":
        find_sub()
    if len(st.session_state.sub) >= 1:
        show_image_3(st.session_state.list_image,st.session_state.sub)
    