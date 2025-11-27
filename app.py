""" Flask实现类 """
import argparse
import json
import logging
import os

from flask import Flask, request, render_template, jsonify

from chatbot import ChatBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str, default=r'Qwen1.5-7B-Chat')
    args = parser.parse_args()
    logger.info(f'开始加载对话模型：{os.path.basename(args.model_name)}...')
    chat_bot = ChatBot(model_name=args.model_name)
    logger.info(f'对话模型加载到设备：{chat_bot.device}')
    logger.info('模型加载成功！')
except Exception as e:
    logger.error(f'加载模型失败：{e}')
    chat_bot = None

@app.route('/')
def index():
    """
    首页渲染

    :return: 首页样式
    """
    return render_template('index.html')

@app.route('/chatbot/reply', methods=['POST'])
def reply():
    """
    聊天机器人回复

    :return: 回复结果
    """
    try:
        data = json.loads(request.data)
        message = data['message']
        conversation_id = data['conversation_id']
        logger.info(f'用户输入：{message}')
        logger.info(f"用户组id：{conversation_id}")

        response, conversation = chat_bot.chat(message, conversation_id)

        logger.info(f'机器人回复：{response}')
        return jsonify({'status': 'success', 'reply': response})
    except Exception as e:
        logger.error(f'回复失败：{e}')
        return jsonify({'status': 'fail' ,'error': '处理发生异常'}), 500


@app.route('/chatbot/clear_history', methods=['DELETE'])
def clear_history():
    try:
        data = json.loads(request.data)
        conversion_id = data['conversation_id']

        chat_bot.clear_history(conversion_id)

        logger.info(f"用户组id：{conversion_id}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f'清理历史会话失败：{e}')
        return jsonify({'status': 'fail' ,'error': '处理发生异常'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
