""" 聊天实现类 """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatBot:
    def __init__(self, model_name='Qwen/Qwen1.5-7B-Chat', max_conv_num=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.conversations = {}
        self.max_conv_num = max_conv_num

    def chat(self, user_input, conversation_id='default'):
        # 如果conversation为空，则添加提示词
        default_conversation = [{'role': 'system', 'content': '你是一个非常有用的AI助手！'}]
        conversation = self.conversations.get(conversation_id, default_conversation)

        # 初始化消息，包含提示词和用户输入
        conversation.append({'role': 'user', 'content': user_input})

        # 将对话信息dict转换为字符串形式
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 词编码获取模型输入信息
        model_inputs = self.tokenizer([text], return_tensors='pt').to(self.device)

        # 获取模型输出，对于文本生成类需要使用.generate方法
        generate_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        # 返回的结果包含输入和输出两部分，只获取新生成的输出
        generate_ids = [
            outputs[len(inputs):] for inputs, outputs in zip(model_inputs.input_ids, generate_ids)
        ]

        # 将返回的结果按批进行解码
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        # 将结果添加到对话中
        conversation.append({'role': 'assistant', 'content': response})

        # 如果会话超过支持的最大数量，则取最后几轮会话
        max_messages = 2 * self.max_conv_num
        if len(conversation) > max_messages + 1:
            conversation = [conversation[0]] + conversation[-max_messages:]

        return response, conversation

    def clear_history(self, conversation_id):
        # 将conversation_id对应的会话置为空
        self.conversations.setdefault(conversation_id, [])