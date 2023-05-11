# MkDocs: Macros plugin

MkDocs, by itself, does not support macros or any form of content reuse across
pages. However, with the help of plugins and extensions, you can introduce this
functionality.

One such plugin is the `mkdocs-macros-plugin`. This plugin allows you to define
variables, simple functions (macros), or even complex Python scripts, and use
them in your Markdown files.

Here is a simple guide on how to install and use this plugin:

1. **Install the plugin**

   You can install the plugin via pip:

   ```bash
   pip install mkdocs-macros-plugin
   ```

2. **Enable the plugin**

   Open your `mkdocs.yml` file and add an entry for the `macros` plugin in
   the `plugins` section:

   ```yaml
   plugins:
     - search
     - macros
   ```

   **Note:** If you already have a `plugins` section in your `mkdocs.yml` file,
   just add `macros` to the list.

3. **Define your macros**

   Create a Python script named `main.py` in the same directory as
   your `mkdocs.yml` file. In this script, define a function for each macro you
   want to use.

   For example:

   ```python
   def define_env(env):
       @env.macro
       def hello(name):
           return f"Hello, {name}!"
   ```

   This script defines a `hello` macro that takes a `name` parameter and returns
   a greeting.

4. **Use your macros**

   Now, you can use the macros you defined in any of your Markdown files. Use
   the `{{ }}` syntax to call your macro.

   For example:

   ```markdown
   {{ hello('World') }}
   ```

   When you build your documentation, this will be replaced with "Hello,
   World!".

Remember to rebuild your documentation (`mkdocs build`) after making these
changes to see the effect of the macros.

Please note that while macros can be very powerful, they also make your
documentation more complex. Use them sparingly and only when necessary.