# MELD Automated Builds

We have an automated build system in place for MELD.
It automatically builds and creates conda packages
for MELD. We are using travis-ci as our build
platform. All build results are available at
<https://travis-ci.org/maccallumlab>.

There are two sets of builds:
- The ones in this directory, which are for testing
- The ones in the [meld-pkg](https://github.com/maccallumlab/meld-pkg) repository,
  which are releases

## Conda Packages

We use the [anaconda](http://anaconda.org) package
manager. The builds can be found on the
[maccallum_lab](http://anaconda.org/maccallum_lab)
anaconda channel.

To use these packages, you must first add a channel and then
load the appropriate package. For example:
```
conda config --add channels maccallum_lab omnia
conda install meld-cuda{VER}
```
Where `VER` can be:
- `75`
- `80`
- `85`
- `90`
- `92`

## Creating a new release

1. Make sure that `CHANGELOG.md` is up to date.
2. Make sure that all changes are committed.
3. Run the following command to update the version number:
    - `bumpversion patch`
4. Push the changes to the `maccallumlab/meld` github repository.
5. Push the tags up as well:
    - `git push --tags`
6. Check to make sure that the new tag shows up as a
   [release](https://github.com/maccallumlab/meld/releases).
7. Edit the `meld-pkg/meld/meta.yaml` file in the
   [meld-pkg](https://github.com/maccallumlab/meld-pkg)
   repository to use the new version.