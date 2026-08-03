from hallucination_agent import ragas_evaluate

result = ragas_evaluate(

    question="What is Amazon S3?",

    answer="Amazon S3 is an object storage service by AWS.",

    reference_answer="Amazon S3 is a scalable object storage service provided by Amazon Web Services.",

    contexts=[
        "Amazon S3 is an object storage service offered by Amazon Web Services.",
        "It provides scalable, secure and durable cloud storage."
    ]
)

print(result)