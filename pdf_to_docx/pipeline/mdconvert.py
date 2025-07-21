from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


class TextToMarkdownAI:
    """
    A class that converts plain text files to structured Markdown format using OpenAI API.
    """

    def __init__(self, input_path: Path, md_path: Path):
        """
        Initialize the converter with input and output paths.
        """
        self.input_path = input_path
        self.md_path = md_path

        # Load API key from .env file
        config = dotenv_values(".env")
        self.api_key = config.get("API_KEY")

        if not self.api_key:
            raise ValueError("API_KEY not found in .env file.")

        # Initialize OpenAI client with custom base_url
        self.client = OpenAI(
            api_key=self.api_key, base_url="https://api.chatanywhere.tech/v1"
        )

    def read_txt(self) -> str:
        """
        Read content from the input text file.
        """
        return self.input_path.read_text(encoding="utf-8")

    def convert_to_md(self, text: str) -> str:
        """
        Send text to OpenAI and receive structured Markdown.
        """
        response = self.client.chat.completions.create(
            # Define model
            model="gpt-3.5-turbo",
            # Set up prompt
            messages=[
                {
                    "role": "system",
                    "content": "你是一個專門將非結構化文件轉成結構化 Markdown 格式的助理",
                },
                {
                    "role": "user",
                    "content": (
                        "請將以下內容完整轉換為 Markdown，並**保留所有段落與說明文字**，"
                        "同時將內容依結構分層，使用以下標題與條列格式：\n\n"
                        "- 章節標題（如「第五章」）：用 `#`\n"
                        "- 節標題（如「第一節」）：用 `##`\n"
                        "- 條目（如「一、重貼現」）：用 `###`\n"
                        "- 小項（如「（一）額度」）：用 `####`\n"
                        "- 條列細項請用 `1.`、`2.` 這種格式\n\n"
                        "請**不要省略任何段落或句子**，即使是說明性段落也要保留。\n\n"
                        "以下是內容：\n\n"
                        f"{text}"
                    ),
                },
            ],
            # Control temperature
            temperature=0.7,
        )
        return response.choices[0].message.content

    def save_md(self, markdown: str):
        """
        Save the Markdown result to the specified file.
        """
        self.md_path.write_text(markdown, encoding="utf-8")

    def run(self):
        """
        Execute the full conversion pipeline.
        """
        text = self.read_txt()
        markdown = self.convert_to_md(text)
        self.save_md(markdown)
        print(f"Markdown file saved to: {self.md_path}")
