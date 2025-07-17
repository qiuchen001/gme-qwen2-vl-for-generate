from transformers import AutoModel
from transformers.utils.versions import require_version
import sys
import os
import torch
from PIL import Image

# 添加模型目录到Python路径
sys.path.append('gme-Qwen2-VL-7B-Instruct')
from gme_inference import GmeQwen2VL

require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


t2i_prompt = 'Find an image that matches the given text.'
texts = [
    # "What kind of car is this?",
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "女孩",
    "狗",
    "海边",
    "女孩与狗在海边玩耍",
    "晴朗",
    "多云",
    "雨雪天",
]

texts_01 = [
    "中国的首都是哪里？"
]

texts_02 = [
    "北京",
    "上海",
    "广州",
    "深圳",
]



# 使用本地图片路径
images = [
    # 'test_image1.jpg',
    # 'test_image2.jpg',
    # 'Tesla_Cybertruck_damaged_window.jpg',
    # '2024_Tesla_Cybertruck_Foundation_Series,_front_left_(Greenwich).jpg',
    'demo.jpeg',
]






# 用于生成任务的测试
gme = GmeQwen2VL("gme-Qwen2-VL-7B-Instruct")

# e_text = gme.get_text_embeddings(texts=texts_01)
# e_text_02 = gme.get_text_embeddings(texts=texts_02)
# print((e_text * e_text_02).sum(-1))


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "请用一句话描述图片."},
#         ],
#     }
# ]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///mnt/data/ai-ground/dataset/videos/360-camera/after/processed_ad_0a2b9214-305a-4f02-97da-a8b7067171a6.mp4",
                # "video": r"E:\playground\ai\datasets\videos\360-camera\after\processed_ad_00db7bfd-837e-49c9-a1c2-4e6483b4fa0c.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "请输出视频中的关键场景的时间片段，并描述场景内容，以结构化的格式输出。"},
        ],
    }
]

result = gme.generate(messages)
print(result)


# texts = ["这张图片的天气如何？"]
# images = [Image.open("demo.jpeg")]
# result = gme.generate(texts=texts, images=images, max_length=30)
# print(result[0])



# print("初始化GME模型...")
# pretrained_model_name_or_path = "gme-Qwen2-VL-7B-Instruct"
# gme = GmeQwen2VL(pretrained_model_name_or_path)

# with open("gme_model_structure.txt", "w", encoding="utf-8") as f:
#     f.write(str(gme.base))

# import inspect
# with open("model_forward_source.txt", "w", encoding="utf-8") as f:
#     f.write(inspect.getsource(gme.base.forward))


# print("测试文本嵌入...")
# 文本嵌入
# e_text = gme.get_text_embeddings(texts=texts)
# print('文本嵌入形状:', e_text.shape)
# print('文本嵌入前5个值:', e_text[0][:5].tolist())

# print("测试图像嵌入...")
# 图像嵌入
# e_image = gme.get_image_embeddings(images=images)
# e_image = gme.get_image_embeddings(images=images, instruction='天气状况：晴朗、多云、雨雪天，请直接输出图片中对应的天气状况.')
# e_image = gme.get_image_embeddings(images=images, instruction='图片中天气状况如何?')

# print('图像嵌入形状:', e_image.shape)
# print('图像嵌入前5个值:', e_image[0][:5].tolist())

# print("计算相似度...")

# print((e_text * e_image).sum(-1))


# 计算相似度
# similarity = e_text @ e_image.T
# print('相似度矩阵:')
# print(similarity.tolist())

# print("测试融合嵌入...")
# 融合嵌入（同时包含文本和图像）
# e_fused = gme.get_fused_embeddings(texts=texts, images=images)
# print('融合嵌入形状:', e_fused.shape)
# print('融合嵌入前5个值:', e_fused[0][:5].tolist())
# print((e_fused[0] * e_fused[1]).sum())



# 下面是临时注释======================================================================

# e_text_last = gme.get_text_embeddings(texts=texts, pooling_strategy="last_token")
# e_text_mean = gme.get_text_embeddings(texts=texts, pooling_strategy="mean_pooling")
# e_text_max = gme.get_text_embeddings(texts=texts, pooling_strategy="max_pooling")
# e_text_first = gme.get_text_embeddings(texts=texts, pooling_strategy="first_token")

# print("Last token shape:", e_text_last.shape)
# print("Mean pooling shape:", e_text_mean.shape)
# print("Max pooling shape:", e_text_max.shape)
# print("First token shape:", e_text_first.shape)

# # # Compare similarities
# # similarity_last_mean = torch.cosine_similarity(e_text_last, e_text_mean, dim=1)
# # similarity_last_max = torch.cosine_similarity(e_text_last, e_text_max, dim=1)
# # similarity_last_first = torch.cosine_similarity(e_text_last, e_text_first, dim=1)

# # print("Cosine similarity between last token and mean pooling:", similarity_last_mean.tolist())
# # print("Cosine similarity between last token and max pooling:", similarity_last_max.tolist())
# # print("Cosine similarity between last token and first token:", similarity_last_first.tolist())


# e_image_last = gme.get_image_embeddings(images=images, pooling_strategy="last_token")
# e_image_mean = gme.get_image_embeddings(images=images, pooling_strategy="mean_pooling")
# e_image_max = gme.get_image_embeddings(images=images, pooling_strategy="max_pooling")
# e_image_first = gme.get_image_embeddings(images=images, pooling_strategy="first_token")


# similarity_last = torch.cosine_similarity(e_text_last, e_image_last, dim=1)
# similarity_mean = torch.cosine_similarity(e_text_mean, e_image_mean, dim=1)
# similarity_max = torch.cosine_similarity(e_text_max, e_image_max, dim=1)
# similarity_first = torch.cosine_similarity(e_text_first, e_image_first, dim=1)

# print("Cosine similarity last token:", similarity_last.tolist())
# print("Cosine similarity mean pooling:", similarity_mean.tolist())
# print("Cosine similarity max pooling:", similarity_max.tolist())
# print("Cosine similarity first token:", similarity_first.tolist())