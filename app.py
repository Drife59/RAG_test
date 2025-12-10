import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document

from answer import answer_question

load_dotenv(override=True)


def format_context(context: list[Document]) -> str:
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result

def get_content_text_from_history_message(history_message: dict) -> str:
    """
        As of gradio 6.X, 'content' is not a str anymore but a dict.
        This function allow to mimic previous version behavior by extracting the text,
        while beeing still compatible with legacy behavior.
    """
    # legacy behavior
    if isinstance(history_message["content"], str):
        return history_message["content"]
    # New behavior (gradio 6+)
    # 'content': [{'text': 'Tell me about insurLLM', 'type': 'text'}]
    return history_message["content"][0]['text']

def chat(history: list[dict]):
    last_message: str = get_content_text_from_history_message(history[-1])

    # As of gradio 6.X, 'content' is not a str anymore but a dict, full pattern is like following:
    # [{'role': 'user', 'metadata': None, 'content': [{'text': 'WESH', 'type': 'text'}], 'options': None}]
    prior =[{"role": "user", "content": get_content_text_from_history_message(m)} for m in history[:-1]]

    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message: str, history: list[dict]):   
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Expert Assistant", fill_height=True) as ui:
        gr.Markdown("# 🏢  Expert Assistant\n droit du travail !")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=450
                )
                message = gr.Textbox(
                    label="Ta question",
                    placeholder="Demande la lune !",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=450,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
