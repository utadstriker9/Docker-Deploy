# Use Jupyter Docker Data Science image
FROM jupyter/datascience-notebook:python-3.10

# Set to root user
USER root

# Copy requirements.txt from build context
COPY requirements.txt /tmp/requirements.txt

# Update Jupyter and kernel
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose Jupyter Notebook on port 8981
EXPOSE 8981

# Set entrypoint with proper arguments
CMD ["start-notebook.sh", , "--NotebookApp.allow_origin='*'", "--NotebookApp.allow_remote_access=True", "--NotebookApp.disable_check_xsrf=True", "--NotebookApp.disable_check_origin=True", "--allow-root", "--NotebookApp.websocket_compression=True"]
