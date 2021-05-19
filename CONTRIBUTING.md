# Contributing

Contributions to MELD are welcome. The preferred approach is to fork and submit a pull request.

Go to the `maccallumlab/meld` github repo and click fork. Then clone your fork to your computer. From inside that repo, run
```
git remote add upstream https://github.com/maccallumlab/meld.git
```

Then you can make commits to your local repo. You can pull in any changes from the main branch with `git merge upstream/master`. Once your work is ready, submit a pull request against `maccallumlab/master` through the github interface.

## Documentation

Scientific software projects often have poor documentation, and MELD is no exception. As a development
team we are working towards improving this situation. All new features should be well documented and
efforts should be made to improve the documentation at every opportunity.

We will follow the [Grand Unified Theory of Documentation](https://documentation.divio.com), where our
documentation is broken into:
- Tutorials
  - Tutorials are lessons that guide the user through a series of steps to finish a project of some
    kind. This is where we show a beginner what they can do with MELD.
  - These are learning oriented. You are the teacher and are responsible for what the student will do.
- How-to Guides
  - How-to guides take the reader through a series of steps to solve a real-world problem.
  - These are recipes to achieve a desired outcome.
  - They are goal oriented.
- Explanation
  - Explanations clarify and illuinate a particular topic.
  - They are understanding oriented.
- Reference
  - Are technical descriptions of the machinery and how to operate it.
  - In MELD, these come from the documentation in the source code itself.

## Style

We use [black](https://github.com/ambv/black) for automatic code formatting. Just let it do its thing and live with the output.

We have settled on the
[google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
format for docstrings. All new code should have full docstrings using this convention.
The existing docstrings are a mix of formats and need to be updated.
There is an
[open issue](https://github.com/maccallumlab/meld/issues/48) for this.

## Testing

We strive to have as much test coverage as possible, although this is hard as the CI severs do not currently support running GPU code.
New code should have tests whenever feasible. To run the tests, use `python -m unittest discover`.

We are also adding python type annotations to the code base. All new code should have type annotations and we aim to add them to existing
code over time. To run the type checks, use `mypy meld`.

## Tagging a new release

To tag a version for release, do the following:
- Commit all changes.
- Run the test suite.
- Update `CHANGELOG.md` to reflect the changes.
- Run `bumpversion patch` to increment to the next version.
- Push the changes:
  - `git push`
  - `git push --tags`
