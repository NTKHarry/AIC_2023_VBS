from groq import Groq

apikey = "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"

client = Groq(api_key=apikey)

def generate(question):
    completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            
            "role": "user",
            "content": f"""
            Translate this sentence: {question} \n
            into English, and answer carefully. \n
            Do not use phrases like 'be translated to' or similar phrases.\n
            The answer must end with the following format:
            Result: <result>

            Example:
            Công an
            Result: Police

            """
        },
    ],
    temperature=0.01,
    max_tokens=126,
    top_p=1,
    stream=False,
    stop=None,)
    return completion.choices[0].message.content.split(':')[-1].split('\n')[0].strip()