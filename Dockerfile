FROM jupyter/scipy-notebook

WORKDIR /home/jovyan

COPY . .

RUN pip install flask pandas scikit-learn numpy requests

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
