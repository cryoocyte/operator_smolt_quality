## operator_template_py

_TODO_REPLACE_IT_

### Operator-related information

* _TODO_REPLACE_IT_

### Local development

* clone repository
* `cp dist.env .env`
* edit `.env` file
* export environment variables from .env file
    * [JetBrains plugin](https://plugins.jetbrains.com/plugin/7861-envfile)
    * export $(grep -v '^#' .env | xargs -0)
* Create [venv](https://docs.python.org/3/library/venv.html)
    * python -m venv .venv
    * source .venv/bin/activate
* pip install -r requirements.txt
* python .

### Local docker build

* clone repository
* cp dist.env .env
* edit `.env` file
* `docker build -t operator_template_py .`
* docker run --rm --env-file .env -v `pwd`:/app -w /app -it operator_template_py:latest 
