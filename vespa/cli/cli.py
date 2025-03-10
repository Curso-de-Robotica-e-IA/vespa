from cyclopts import App

from vespa import vespa as vp

app = App(description='Vespa CLI - Command Line Interface for AI Tasks')


@app.command()
def train(task: str, model: str, dataset: str, format: str):
    """Train a model with a specified dataset and format."""
    vp.set_task(task)
    vp.set_dataset_format(format)
    vp.set_model(model)
    result = vp.train(dataset)
    print('Training Result:', result)


@app.command()
def predict(task: str, model: str, input: str):
    """Run inference on an input image using a selected model."""
    vp.set_task(task)
    vp.set_model(model)
    output = vp.predict(input)
    print('Prediction Output:', output)


@app.command()
def list_models():
    """Display the available models for different tasks."""
    models = vp.list_models()
    print('Available Models:', models)


@app.command()
def list_dataset_formats():
    """Show the supported dataset formats."""
    formats = vp.list_dataset_formats()
    print('Supported Dataset Formats:', formats)


if __name__ == '__main__':
    app.run()
