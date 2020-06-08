# Contributing to Pew Analytics

<!-- This CONTRIBUTING.md is adapted from https://gist.github.com/peterdesmet/e90a1b0dc17af6c12daf6e8b2f044e7c -->

[repo]: https://github.com/pewresearch/pewanalytics
[issues]: https://github.com/pewresearch/pewanalytics/issues
[new_issue]: https://github.com/pewresearch/pewanalytics/issues/new
[email]: info@pewresearch.org

## How you can contribute

There are several ways you can contribute to this project. If you want to know more about why and how to contribute to open source projects like this one, see this [Open Source Guide](https://opensource.guide/how-to-contribute/).

### Share the love ❤️

Think **pewanalytics** is useful? Let others discover it, by telling them in person, via Twitter or a blog post.

### Ask a question ⁉️

Using **pewanalytics** and got stuck? Check out the [documentation](https://pewresearch.github.io/pewanalytics/). 
Still stuck? Post your question as an [issue on GitHub][new_issue]. While we cannot offer user support, we'll try to do our best to address it, as questions often lead to better documentation or the discovery of bugs.

Want to ask a question in private? Contact the package maintainer by [email][email].

### Propose an idea 💡

Have an idea for a new **pewanalytics** feature? Take a look at the [issue list][issues] to see if it isn't included or suggested yet. If not, suggest your idea as an [issue on GitHub][new_issue]. While we can't promise to implement your idea, it helps to:

* Explain in detail how it would work.
* Keep the scope as narrow as possible.

See below if you want to contribute code for your idea as well.

### Report a bug 🐛

Using **pewanalytics** and discovered a bug? That's annoying! Don't let others have the same experience and report it as an [issue on GitHub][new_issue] so we can fix it. A good bug report makes it easier for us to do so, so please include:

* Your operating system name and version (e.g. macOS 10.13.6).
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Contribute code 📝

Care to fix bugs or implement new functionality for **pewanalytics**? Awesome! 👏 Have a look at the [issue list][issues] and leave a comment on the things you want to work on. When making contributions, please follow the development guidelines below.

#### Development guidelines

We try to follow the [GitHub flow](https://guides.github.com/introduction/flow/) for development, and we use Python docstrings and [Sphinx](https://www.sphinx-doc.org/en/master/) to document all of our code. 

1. Fork [this repo][repo] and clone it to your computer. To learn more about this process, see [this guide](https://guides.github.com/activities/forking/).
2. If you have forked and cloned the project before and it has been a while since you worked on it, [pull changes from the original repo](https://help.github.com/articles/merging-an-upstream-repository-into-your-fork/) to your clone by using `git pull upstream master`.
3. Make your changes:
    * Write your code.
    * Test your code (bonus points for adding unit tests).
    * Document your code (see function documentation above).
4. If you added unit tests, make sure everything works by running the `python -m unittest tests` command from the root directory of the repository. 
5. If you added or updated documentation, build a fresh version of the docs by running the `make github` command from the root directory of the repository.
6. Commit and push your changes.
7. Submit a [pull request](https://guides.github.com/activities/forking/#making-a-pull-request).