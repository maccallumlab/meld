# Contributing

Contributions to MELD are welcome. The preferred approach is to fork and submit a pull request.

Go to the `maccallumlab/meld` github repo and click fork. Then clone your fork to your computer. From inside that repo, run
```
git remote add upstream https://github.com/maccallumlab/meld.git
```

Then you can make commits to your local repo. You can pull in any changes from the main branch with `git merge upstream/master`. Once your work is ready, submit a pull request against `maccallumlab/master` through the github interface.

## Style

We use [black](https://github.com/ambv/black) for automatic code formatting. Just let it do its thing and live with the output.

We have settled on the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) format for docstrings. All new code should have full
docstrings using this convention. The existing docstrings are a mix of formats and need to be updated. There is an
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
