In this directory, you will find Marimo notebooks that cannot (yet?) be compiled to WASM and run in the browser. These notebooks are meant to be run in a local environment. There are several ways to get started using Marimo locally. Go here to find out the best option for you:

https://docs.marimo.io/getting_started/installation/ 


Once you have Marimo installed, you can run the notebooks in this directory by navigating to the directory in your terminal and running:

```bash
marimo edit notebook_name.py --sandbox
```

This will open the notebook in your default web browser, where you can interact with it and run the cells. You can also run the notebook 'app mode' by running:

```bash
marimo run notebook_name.py --sandbox
```

This will hide the code cells so it feels more like a traditional app. If you open the app in edit mode, you can still toggle between app mode and edit mode via a button in the right sidebar.
