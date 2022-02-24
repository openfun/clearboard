FROM public.ecr.aws/lambda/python:3.9 as core

# Copy sources
# LAMBDA_TASK_ROOT is defined in base image and its value is /var/task
COPY lambda-enhance/ "${LAMBDA_TASK_ROOT}/lambda-enhance"

RUN pip3 install --upgrade pip

# Install the function's dependencies using file requirements.txt
# from your project folder.
RUN  pip3 install -r "${LAMBDA_TASK_ROOT}/lambda-enhance/requirements.txt" --target "${LAMBDA_TASK_ROOT}/lambda-enhance"

CMD ["lambda-enhance.app.handler"]
