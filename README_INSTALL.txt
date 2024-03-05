git clone https://github.com/yuragorlo/flai.git
cd flai
python -m venv env
source env/bin/activate
pip3 install -r requirements.txt
cd RestGPT
nano config.yaml # add your openai_api_key and flaidata_token to config.yaml
python3 run.py
select flai