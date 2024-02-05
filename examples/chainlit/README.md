# Running the chainlit apps

In your Python virtual env, ensure you have 
installed `langroid` with the `chainlit` extra using, e.g.

```bash
pip install langroid[chainlit]
```

Or if you already have `langroid` installed, you can install the `chainlit` extra using:

```bash
pip install chainlit
```

To check that `chainlit` is installed, run:

```bash
chainlit hello
```

and you should see the `hello app` open in your browser.

## Configuration

You can configure some aspects of the chainlit app via these files,
which are included in this repo at the root level (see
the Chainlit [customization docs](https://docs.chainlit.io/customisation/overview) for more details):
- `.chainlit/config.toml` to customize project, features, UI (see [here](https://docs.chainlit.io/backend/config/overview))
- `.chainlit/translations/en-US.json` for various ["translations"](https://docs.chainlit.io/customisation/translation) and language-specific
   customizations. In particular, the default text in the input box is customized here.
- `chainlit.md`, which contains the initial "Readme" content

