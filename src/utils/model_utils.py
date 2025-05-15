from openai import OpenAI


def return_model(model_config):

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{model_config.port_num}/v1"
    )

    return client