import requests as rq
import json
import xml.etree.ElementTree as ET
import feedparser
import numpy as np
import hashlib
import openai
import sys
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def send_email(subject, message):
    # 邮件配置
    smtp_server = 'smtp.gmail.com'  # SMTP服务器地址
    smtp_port = 587  # SMTP端口号
    smtp_username = 'sun889999@gmail.com'  # 发件人邮箱用户名
    smtp_password = 'hobptivsdezalige'  # 发件人邮箱密码

    # 构建邮件
    from_email = 'sun889999@gmail.com'  # 发件人邮箱
    to_email = 'sun889999.pppp@blogger.com'  # 收件人邮箱

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # 如果要附加文件，请使用以下代码
    # with open('example.txt', 'rb') as attachment:
    #     part = MIMEApplication(attachment.read())
    #     part.add_header('Content-Disposition', 'attachment', filename='example.txt')
    #     msg.attach(part)

    # 连接到SMTP服务器并发送邮件
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 使用TLS加密连接
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print('邮件发送成功')
    except Exception as e:
        print(f'邮件发送失败: {str(e)}')

def get_article_content(url):
    # 发送HTTP请求获取网页内容
    response = rq.get(url)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找所有 <script type="application/ld+json"> 元素
        script_tags = soup.find_all('script', type='application/ld+json')

        # 遍历每个 <script> 元素并获取其内容
        for script_tag in script_tags:
            # 解析 JSON 数据
            try:
                data = json.loads(script_tag.string)
                return data
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
    else:
        print(f"无法访问网页，状态码: {response.status_code}")

def gen_ai_result(prompt):
    # 设置您的API密钥
    api_key = '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'

    # 初始化OpenAI客户端
    openai.api_key = api_key

    # 发送请求生成文本
    response = openai.Completion.create(
        engine="text-davinci-003",  # 指定引擎，可以是不同的模型
        prompt=prompt,
        max_tokens=1000  # 生成的最大标记数
    )
    return response.choices[0].text.strip()


def cosine_similarity(vector1, vector2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vector1: numpy数组，第一个向量
    vector2: numpy数组，第二个向量

    返回：
    float，余弦相似度值
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# 文字專向量
def request_open_ai(text):
    # print(text)
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"  # 替换为你的OpenAI API密钥
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = rq.post(url, headers=headers, json=data)
    result = response.json()
    return result['data'][0]['embedding'] # 向量資料




api_link = "https://news.pts.org.tw/xml/newsfeed.xml" # 公視rss新聞
# api_link = "https://about.pts.org.tw/rss/XML/newsletter.xml" # 公視rss新聞稿
NewsFeed = feedparser.parse(api_link)
rss_datas = []
for i in range(len(NewsFeed.entries)):
    entry = NewsFeed.entries[i]
    rss_datas.append({'title': entry.title, 'link': entry.link})

# 使用餘弦定理算出相似度
# 使用示例
open_datas = []
for i in range(len(rss_datas)):
    embedding = request_open_ai(rss_datas[i]['title'])
    open_datas.append({'title': rss_datas[i]['title'], 'embedding':embedding, 'link': rss_datas[i]['link']})

score_datas = {}
match_key = 0

for i in range(len(open_datas)):
    for x in range(len(open_datas)):
        if open_datas[x]['title'] not in score_datas:
            embedding_vector1 = open_datas[i]['embedding']
            embedding_vector2 = open_datas[x]['embedding']
            similarity_score = cosine_similarity(embedding_vector1, embedding_vector2) # 比較標題相似性
            if similarity_score >= 0.9 and similarity_score < 0.9999999999999999:
                # 判斷該score_datas是否有該i的title
                if open_datas[i]['title'] not in score_datas:
                    # 若無則宣告
                    score_datas[open_datas[i]['title']] = {'data':{}, 'match_datas':[]}
                    score_datas[open_datas[i]['title']]['data'] = open_datas[i]
                # 把該x資料加入到score_datas
                score_datas[open_datas[i]['title']]['match_datas'].append(open_datas[x])

new_datas = []
for key, value in score_datas.items():
    page_json_i = get_article_content(value['data']['link'])
    description = page_json_i['description']
    datas_source = "資料來源：" + value['data']['link'] + "(" + value['data']['title']  + ")"
    for x in range(len(value['match_datas'])):
        page_json_x = get_article_content(value['match_datas'][x]['link'])
        description +=  page_json_x['description']
        datas_source += "資料來源：" + value['match_datas'][x]['link'] + "(" + value['match_datas'][x]['title']  + ")"
    detail = gen_ai_result("把以下內容整理為約300字以內的新聞文章「" + description + "」")
    kayword = gen_ai_result("依照以下內容整理出5個以下的關鍵詞「" + detail + "」")
    title = "[ 每日AI讀報 ] " + gen_ai_result("把以下內容下一個新聞標題「" + detail + "」")
    detail += "關鍵字：" + kayword
    detail += datas_source
    send_email(title, detail)

