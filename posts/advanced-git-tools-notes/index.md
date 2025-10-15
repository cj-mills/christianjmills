---
categories:
- git
- notes
date: '2021-12-29'
description: My notes from Tobias Gunther's video covering advanced git tools.
hide: false
search_exclude: false
title: Notes on Advanced Git Tools


twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: /images/default-preview-image-black.png
open-graph:
  image: /images/default-preview-image-black.png

aliases:
- /Notes-on-Advanced-Git-Tools/
---

* [Overview](#overview)
* [Interactive Rebase](#interactive-rebase)
* [Cherry-Picking](#cherry-picking)
* [Reflog](#reflog)
* [Submodules](#submodules)
* [Search and Find](#search-and-find)
* [Additional Resources](#additional-resources)



## Overview

Here are some notes I took while watching Tobias Gunther's [video](https://www.youtube.com/watch?v=qsTthZi23VE) covering advanced git tools such as interactive rebase, cherry-picking, reflog, submodules, and search and find.



## Interactive Rebase

- The swiss army knife of git commands
- A tool for optimizing and cleaning up your commit history
    - Change a commit’s message
    - Delete commits
    - Reorder commits
    - Combine multiple commits into one
    - Edit/split an existing commit into multiple new ones
- Warning Note
    - Interactive rebase rewrites your commit history
    - The commits you manipulate will have new hash ID’s
    - You should not use interactive rebase on stuff you already pushed to a remote repository
- Use it for cleaning up your local commit history before merging it into a shard team branch
    - Example: When you are done developing on a feature branch
- Steps
    1. What should be the “base” commit?
        - At least the parent commit of the one you want to manipulate
    2. `git rebase -i HEAD~3`
    3. In the editor, only determine which *actions* you want to perform. Don’t change commit data in this step.
- Change commit message
    - Most recent commit
        - `git commit ammend`
    - Older commits
        - `git rebase -i HEAD~<number-of-commits-before-HEAD>`
            - Opens editor window with commits in selected range
        - Mark up the line with the target commit with the desired action
            - `reword <commit-hash> <commit-message>`
        - Save and close editor
            - New editor window will open
        - Change commit message
        - Save and close editor
- Combine two commits
    - Determine the base commit
        - `git rebase -i HEAD~<number-of-commits-before-HEAD>`
    - Mark up the line with the target commit with the desired action
        - `squash <commit-hash> <commit-message>`
        - Will combine commit in the marked line with the commit in the line above it
    - Save and close editor
        - New editor window will open
    - Add commit message for new commit

## Cherry-Picking

- Integrating single, specific commits
    - Normally, you should integrate commits on the branch level
- Should only be used for special situations
- Moving a commit to a different branch
    - Switch to the branch the commit will be moved to
        - `git switch <branch-name>`
    - `git cherry-pick <commit-hash>`
    - Clean up branch the commit was moved from
        - `git switch <commit-origin-branch>`
        - `git reset --hard HEAD~1`

## Reflog

- journal where git logs every movement of the HEAD pointer
- Recovering Deleted Commits
    - Delete commit
        - `git reset --hard <commit-hash>`
    - Open reflog
        - `git reflog`
        - entries are ordered chronologically with the most recent at the top
    - Restore state before deletion
        - `git branch <new-branch> <commit-hash-before-deletion>`
- Recovering deleted branches
    - Delete branch
        - `git branch -d <branch-name>`
    - Open reflog
        - `git reflog`
            - Find the commit hash before the deletion
    - Restore branch
        - `git branch <branch-name> <commit-hash>`
    

## Submodules

- A standard git repository that is nested inside another repository
    - Don’t manually copy-paste third-party code
        - mixes external code with your own files
        - Updating the external code is a manual process
- The actual content of a submodule is not part of the parent git repository
    - stored in a `.gitmodules` file, `.gitconfig` file and `.git/.gitmodules` file
        - Remote URL
        - Local path
        - Checked out revision
- Adding a submodule
    - Open your git project
    - Create a new folder for the submodule (e.g. lib)
    - Enter new folder
    - `git submodule add <remote-repo-url>`
    - creates a new `.gitmodules` file
- Need to commit to main repository
    - `git commit -m "commit message"`
- Cloning a project with submodules
    - Clone project like normal
        - `git clone <remote-url>`
        - submodule folders are empty by default
    - Initialize submodules
        - `git submodule update --init --recursize`
        - triggers cloning processes
- Cloning a project with submodules (single step)
    - `git clone --recurse-submodules <remote-repo-url>`
- Check out revisions in submodule
    - submodule repositories are checked out on a commit, not a branch

## Search and Find

- Filtering your commit history
    - can use these in combination
    - by date `--before/--after`
        - `git log --after="2021-7-1" --before="2021-7-5"`
    - by message `--grep`
        - `git log --grep="search-string"`
        - supports regex
    - by author `--author`
        - `git log --author="author-name"`
    - by file `-- <filename>`
        - `git log -- <filename>`
        - the `--`   is to make sure it is not confused for a branch name
    - by branch `<branch-name>`
        - `git log <branch-name>`
    - commits in one branch but not another one
        - `git log <branch-A>..<branch-B>`
    
    

### Additional Resources

[Advanced Git Kit](https://www.git-tower.com/learn/git/advanced-git-kit/)




**References:**

* [Advanced Git Tutorial - Interactive Rebase, Cherry-Picking, Reflog, Submodules and more](https://www.youtube.com/watch?v=qsTthZi23VE)






{{< include /_about-author-cta.qmd >}}
