
!!! note
   Caveat Lector. May not be fully accurate. Trust but Verify!

# Migrating from Ghost To Hugo

Migrating your Ghost blog to an open-source Hugo site requires several steps,
including exporting your content from Ghost, installing Hugo and setting up a
new theme, and converting your content to Hugo-compatible Markdown format. Here
is a step-by-step guide to help you through the process:

1. Export your content from Ghost:
   a. Log in to your Ghost admin panel.
   b. Navigate to the "Labs" section.
   c. Under the "Export your content" header, click the "Export" button to
   download a JSON file containing your content.

2. Install Hugo:
   a. Visit the official Hugo website (https://gohugo.io/) and follow the
   installation instructions for your operating system.
   b. Once Hugo is installed, verify the installation by running `hugo version`
   in the command line.

3. Create a new Hugo site:
   a. Run `hugo new site my-new-hugo-site` to create a new Hugo site in a
   directory named "my-new-hugo-site."
   b. Change your working directory to the new site's directory
   using `cd my-new-hugo-site`.

4. Set up a theme for your Hugo site:
   a. Choose a Hugo theme from the official themes
   repository (https://themes.gohugo.io/).
   b. Follow the theme's installation instructions, which typically involve
   cloning the theme's repository into the "themes" directory of your new Hugo
   site.

5. Convert Ghost content to Hugo-compatible Markdown format:
   a. You can use a tool like "
   GhostToHugo" (https://github.com/jbarone/ghostToHugo) to convert your Ghost
   JSON export to Hugo-compatible Markdown files.
   b. Follow the tool's instructions to convert your JSON file and save the
   resulting Markdown files in the "content" directory of your Hugo site.

6. Configure your new Hugo site:
   a. Edit the "config.toml" or "config.yaml" file in the root of your Hugo site
   to configure site-wide settings, such as site title, author name, and theme.
   b. Customize the theme by following the theme's documentation.

7. Test your new Hugo site locally:
   a. Run `hugo server` in the command line to start a local web server.
   b. Visit http://localhost:1313/ to preview your Hugo site. The site will
   automatically update as you make changes to the content or configuration.

8. Deploy your Hugo site:
   a. Choose a hosting service, such as Netlify, GitHub Pages, or GitLab Pages,
   and follow their documentation for deploying a Hugo site.
   b. Update your DNS settings to point your domain name to your new Hugo site.

9. Redirects and SEO considerations:
   a. If your Ghost blog URLs differ from your new Hugo site, set up 301
   redirects to maintain search engine rankings and avoid broken links.
   b. Ensure that you have configured the Hugo site's metadata, such as meta
   tags and Open Graph tags, to optimize for search engines and social sharing.

Once you've completed these steps, you should have successfully migrated your
Ghost blog to a Hugo site.

# Alternatives to Hugo

There are several free and open-source alternatives to Hugo for creating
static websites and blogs. Here are some popular options:

1. Jekyll (https://jekyllrb.com/): A popular static site generator built with
   Ruby. Jekyll is known for its simplicity and seamless integration with GitHub
   Pages.

2. Gatsby (https://www.gatsbyjs.com/): A React-based, open-source framework for
   building fast, modern websites and applications. Gatsby provides excellent
   performance and a rich plugin ecosystem.

3. Next.js (https://nextjs.org/): A powerful React-based framework that supports
   static site generation, server-side rendering, and API routes. Next.js is
   suitable for building scalable, high-performance websites and applications.

4. Eleventy (https://www.11ty.dev/): A simpler and more flexible static site
   generator built with JavaScript. Eleventy is known for its speed and ease of
   use, supporting multiple template languages.

5. Hexo (https://hexo.io/): A fast, simple, and powerful blog framework powered
   by Node.js. Hexo is particularly popular among developers who want to create
   and deploy their blogs quickly.

6. Pelican (https://blog.getpelican.com/): A static site generator written in
   Python. Pelican supports Markdown, reStructuredText, and AsciiDoc content and
   is easy to customize using themes and plugins.

7. Gridsome (https://gridsome.org/): A Vue.js-powered, modern website generator
   for building fast, optimized websites. Gridsome includes a powerful data
   layer and is optimized for search engines and social sharing.

All these alternatives offer free, open-source solutions for building static
websites and blogs. Depending on your preferred technology stack and the
specific features you require, one of these options may be better suited to your
needs than Hugo. Additionally, you can host your static site for free on
platforms like GitHub Pages, GitLab Pages, Netlify, or Vercel.

# Math notation support

Most static site generators can support MathJax or other math notation rendering
libraries, such as KaTeX. Here are some popular static site generators, all of
which are free and open-source, that can support math notation with minimal
configuration:

1. Hugo (https://gohugo.io/): As previously mentioned, Hugo is a popular static
   site generator built with Go. To enable MathJax, you can follow these steps:

   a. Choose a Hugo theme that supports MathJax, or add support to your existing
   theme by including the MathJax script in your theme's layout files, usually
   in the "header" or "footer" partials.

   b. You can follow the instructions in the official Hugo documentation for
   using MathJax: https://gohugo.io/content-management/formats/#mathjax

2. Jekyll (https://jekyllrb.com/): Jekyll is another popular static site
   generator built with Ruby. To enable MathJax, follow these steps:

   a. Add the MathJax script to your Jekyll site's layout files, usually in
   the "head" or "footer" include files.

   b. You can follow the instructions in the official Jekyll documentation for
   using MathJax: https://jekyllrb.com/docs/extras/#math-support

3. Gatsby (https://www.gatsbyjs.com/): Gatsby is a React-based, open-source
   framework. To enable math notation, you can use the gatsby-remark-katex
   plugin (https://www.gatsbyjs.com/plugins/gatsby-remark-katex/) or the
   gatsby-plugin-mathjax (https://www.gatsbyjs.com/plugins/gatsby-plugin-mathjax/)
   plugin.

4. Eleventy (https://www.11ty.dev/): Eleventy is a simple and flexible static
   site generator built with JavaScript. You can add MathJax support by
   following this guide: https://skeptic.de/projects/eleventy-mathjax/

These static site generators should meet your requirements for rendering math
notation with MathJax or similar libraries, and they are free to use. To host
your site for free, you can use platforms like GitHub Pages, GitLab Pages,
Netlify, or Vercel.

# Adding Math support to a Ghost theme

Ghost's paid plans include access to a variety of themes, some of which may
support math notation rendering using MathJax or KaTeX out of the box. However,
if you find a theme that you like and it doesn't support math notation by
default, you can easily add support by modifying the theme. Here's how you can
do that:

1. Download the theme you want to use, either from Ghost's
   marketplace (https://ghost.org/marketplace/) or from a third-party provider.

2. Unzip the theme and locate the layout file where you want to include the
   MathJax or KaTeX script. This is usually a file named `default.hbs`
   or `index.hbs` within the theme's directory.

3. Add the appropriate script tag for MathJax or KaTeX to the file. For MathJax,
   you can use:

```html

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js"
        integrity="sha384-2r7rF3vEOA3QzdfjGyzo2auhc//ndTl8W7YpYFJ1t7YFUVKy8PHTDfsPOt0IKIWU"
        crossorigin="anonymous"></script>
```

For KaTeX, you can use:

```html

<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.css"
      integrity="sha384-z1fJDqw8L2z3x4gD3f6D9XvrSvR5hGp6w5c6AB5e5zJxssnmupdW+jB1srz0Xegt"
      crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/katex@0.13.18/dist/katex.min.js"
        integrity="sha384-ppvOvTfWz8ag+ssl52D9xFAj+jUf5MxWFXg5pLw//kCmD1zjbbf8ao5P5dBEY5YU"
        crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/markdown-it-texmath@0.9.0/dist/texmath.min.js"
        integrity="sha384-H42aIbVeKw7VZGz3u3eVyyJjE1s7OF+gkx1YCja/05yvnkxiB7zCZDlYtx9z1BbJ"
        crossorigin="anonymous"></script>
```

4. Save the modified file and compress the theme directory back into a `.zip`
   file.

5. Upload the modified theme to your Ghost blog:
   a. Log in to your Ghost admin panel.
   b. Navigate to the "Design" section.
   c. Click on the "Upload a theme" button and upload your modified theme.

After completing these steps, your chosen Ghost theme should support math
notation rendering using either MathJax or KaTeX.