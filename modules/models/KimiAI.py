from ..utils import *
from .base_model import BaseLLMModel
from ..index_func import *
from ..config import default_chuanhu_assistant_model
from openai import OpenAI

class KimiAIClient(BaseLLMModel):
    def __init__(
        self,
        model_name,
        api_key,
        system_prompt=INITIAL_SYSTEM_PROMPT,
        temperature=0.3,
        top_p=1.0,
        user_name=""
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user=user_name
        )
        self.api_key = api_key
        self.need_api_key = True
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.moonshot.cn/v1",)
        if system_prompt is not None:
            self.system_prompt = system_prompt

    def get_answer_stream_iter(self):
        if not self.api_key:
            raise Exception(NO_APIKEY_MSG)
        response = self._get_response(stream=True)
        if response is not None:
            partial_text = ""
            for chunk in response:
                if not chunk.choices[0].delta.content:
                    continue
                partial_text += chunk.choices[0].delta.content
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        response = self._get_response()
        # response = json.loads(response.text)
        content = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]
        return content, total_token_count

    def _get_response(self, stream=False):
        if not self.api_key:
            raise Exception(NO_APIKEY_MSG)

        system_prompt = self.system_prompt
        history = self.history
        logging.debug(colorama.Fore.YELLOW +
                      f"{history}" + colorama.Fore.RESET)

        if system_prompt is not None and len(system_prompt.strip())>0:
            history = [construct_system(system_prompt), *history]

        response=self.client.chat.completions.create(
            model=self.model_name,
            messages=history,
            stream=stream,
            temperature=self.temperature
        )

        return response

    def count_token(self, user_input):
        input_token_count = count_token(construct_user(user_input))
        if self.system_prompt is not None and len(self.all_token_counts) == 0:
            system_prompt_token_count = count_token(
                construct_system(self.system_prompt)
            )
            return input_token_count + system_prompt_token_count
        return input_token_count

    def summarize_index(self, files, chatbot, language):
        from modules.token_text_spliter_mapping import set_cache_dir_and_change_mapping
        set_cache_dir_and_change_mapping()

        status = gr.Markdown.update()
        if files:
            index = construct_index(None, file_src=files)
            status = i18n("总结完成")
            logging.info(i18n("生成内容总结中……"))
            os.environ["OPENAI_API_KEY"] = self.api_key
            from langchain.callbacks import StdOutCallbackHandler
            from langchain.chains.summarize import load_summarize_chain
            from langchain.prompts import PromptTemplate
            # from langchain_community.llms import OpenAIChat
            from langchain.chat_models import ChatOpenAI

            prompt_template = (
                "Write a concise summary of the following:\n\n{text}\n\nCONCISE SUMMARY IN "
                + language
                + ":"
            )
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            llm = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://api.moonshot.cn/v1",
            )
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                return_intermediate_steps=True,
                map_prompt=PROMPT,
                combine_prompt=PROMPT,
            )
            summary = chain(
                {"input_documents": list(index.docstore.__dict__["_dict"].values())},
                return_only_outputs=True,
            )["output_text"]
            print(i18n("总结") + f": {summary}")
            chatbot.append([i18n("上传了") + str(len(files)) + "个文件", summary])
        return chatbot, status

