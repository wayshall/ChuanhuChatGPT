from ..utils import *
from .base_model import BaseLLMModel
from zhipuai import ZhipuAI


class ZhipuAIClient(BaseLLMModel):
    def __init__(
        self,
        model_name,
        api_key,
        system_prompt=INITIAL_SYSTEM_PROMPT,
        temperature=1.0,
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
        self.client = ZhipuAI(api_key=self.api_key)
        if system_prompt is not None:
            self.system_prompt = system_prompt

    def get_answer_stream_iter(self):
        if not self.api_key:
            raise Exception(NO_APIKEY_MSG)
        response = self._get_response(stream=True)
        if response is not None:
            partial_text = ""
            for chunk in response:
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
            stream=stream
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

