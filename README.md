# meld #

Modeling with limited data

# Version Control #

All development is with [git](http://git-scm.com) and [github](http://github.com).

All development should take place on a separate branch. A branch named branch-name can be created with:

    git checkout -b branch-name

## Commit Messages ##

You can then add and commit as many changes in this private branch. No one will see your changes yet. Your commit
messages should begin with a single line, complete description of what the change was. If needed, further lines can
be added after an initial blank line. Now is not the time to get lazy, it takes 30 extra seconds to have a good commit
message that will help you and others decipher what you did. Here's a good example:

    Fixed bug with energies/structures.

    Previously, the energies and structures were out of sync
    by one MD step. The energies were for the N-1th step, while
    the structure was for the Nth step. This meant that both the
    restraint energies and the Amber energies were out of sync.

    This is now fixed by saving a restart file for the N-1th step
    and then using this instead of the last frame.

    When using a constant collection, the restraint energies reported
    by Amber and by the ladder code should now agree almost exactly.
    These numbers won't necessarily match for other collection types
    because the springs may have been 'flickered' between Amber
    finishing and the ladder code printing the energies.

Here is a bad commit message:

    mods for keeneland

## Uploading Your Branch ##

Once you have developed your new code, you will need to upload it to github:

    git push -u origin branch-name

This will push branch-name up to origin (github) and create a tracking branch (-u).

## Requesting a Code Review ##

Next, sign in to github. Switch to the Laufer Center account and find the meld repository. Switch
to your custom branch. You can then submit a pull request, which will allow others to review your code.
Once they have done so, you can merge your changes into the master branch.

## Cleaning up Your Branches ##

Once your changes are merged, you no longer need the branches, so you should clean up:

    git branch -d branch-name
    git push origin --delete branch-name
    git remote prune origin

This will delete your local branch, delete the branch off the server, and delete any unused local tracking
branches.

## Code Review ##

All code review happens in the github interface. This allows for general questions and answers, as well as
commenting on specific lines of code.

What to look for:

* Are there unit tests? (see below)
* Can you understand what the code does?
* Is it efficient / elegant?
* Are the public methods adequately documented?
* Is the code [pep8](http://www.python.org/dev/peps/pep-0008/) compatible?

## Unit Tests ##

All code should be developed using Test Driven Development. That is, writing tests one at a time, and then
writing just enough code to pass the tests. This is a broad topic, so look at the existing code and then
consult google.

We use two frameworks for testing. Both in the standard library of python 2.7+:

* [unittest](http://docs.python.org/2/library/unittest.html)
* [mock](https://pypi.python.org/pypi/mock)

The following are useful for automatically running tests:

* [nose](https://nose.readthedocs.org/en/latest/)
* [sniffer](https://pypi.python.org/pypi/sniffer)

