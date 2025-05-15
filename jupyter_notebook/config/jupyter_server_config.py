c = get_config()
c.NotebookApp.tornado_settings = {
    "headers": {
        "Content-Security-Policy": "frame-ancestors 'self' *"
    }
}
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.disable_check_xsrf = True
c.NotebookApp.disable_check_origin = True
c.NotebookApp.trust_xheaders = True
c.NotebookApp.websocket_compression = True
c.NotebookApp.notebook_dir=/root/notebooks
