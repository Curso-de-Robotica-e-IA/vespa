import sys

from cyclopts import App

from vespa import vespa as vp


app = App(name='Vespa')


@app.command()
def train(
    task: str = None,
    model: str = None,
    dataset: str = None,
    format: str = None,
):
    """Train a model with a specified dataset and format."""
    if not task or not model or not dataset or not format:
        print('Error: Missing parameters for training.')
        return

    vp.set_task(task)
    vp.set_dataset_format(format)
    vp.set_model(model)
    vp.train(dataset)
    print('Training Result: Success')


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


def main():
    """Entry point for Vespa CLI."""
    if 'pytest' in sys.modules:
        app([])
    else:
        app()


if __name__ == '__main__':
    main()
