ground-truth dataset: ac_num.csv (but only 351 are labeled)

Evaluating LLMs

Run prompt with previous and future contexts <br />
python eval-both-context.py 


Run prompt with future contexts <br />
python eval-future-context.py 


Run prompt with no contexts <br />
python eval-no-context.py 


Run prompt with previous contexts <br />
python eval-previous-context.py 

Sample LLM labelled csv files: 

Llama-3.3-70B-Instruct.csv <br />
Meta-Llama-3-8B-Instruct.csv <br />
Mistral-7B-Instruct-v0.1.csv <br />
Mixtral-8x22B-Instruct-v0.1.csv <br />
Mixtral-8x7B-Instruct-v0.1.csv

message_sampling.py <br />
samples messages from a Telegram database (MongoDB) for annotation


