---
aliases:
- /Notes-on-Version-Control/
categories:
- git
- notes
date: '2021-12-29'
description: My notes from Tobias Gunther's video covering tools and concepts for
  version control with git.
hide: false
layout: post
search_exclude: false
title: Notes on Version Control
toc: false

---

* [Overview](#overview)
* [The Perfect Commit](#the-perfect-commit)
* [Branching Strategies](#branching-strategies)
* [Pull Requests](#pull-requests)
* [Merge Conflicts](#merge-conflicts)
* [Merge versus Rebase](#Merge-versus-rebase)



## Overview

Here are some notes I took while watching Tobias Gunther's [video](https://www.youtube.com/watch?v=Uszj_k0DGsg) covering tools and concepts for mastering version control with git.



## The Perfect Commit

- Add the *right* changes
    - Goal is to make a commit that makes sens
    - Should only contain changes from a single topic
    - Don’t cram everything into single commit
        - Makes it more difficult to what changes were made in retrospect
    - Use Git staging area concept
        - allows you to select specific files or parts of files for the next commit
    - include specific file
        - `git add <filepath>`
    - include part of a file
        - `git add -p <filepath>`
        - steps through every chunk of change in the file and asks whether to add it
- Compose a *good* commit message
    - Subject: concise summary of what happened
        - If you are struggling to keep it short, you might have too many topics in the commit
        - Example: “Add captcha for email signup”
    - Body: more detailed explanation
        - What is now different than before?
        - What is the reason for the change?
        - Is there anything to watch out for or is there anything particularly remarkable?
        - Add an empty line after the subject to let git know that you are now writing the body
        - Example:
            - Email signups now require a captcha to be completed:
                - signup.php uses our captcha library
                - invalid signup attempts are now blocked

## Branching Strategies

- Need to have a clear convention for how your team will work with branches
- Git allows you to create branches, but does not tell your how to use them
- You need a written best practice of how work is ideally structured in your team to avoid mistakes and collisions
- Your branching workflow is highly dependent on your team, size of team, your project, and how you handle releases
- Having a clear convention for branches helps onboard new team members
- Consider your project, release cycle, and team
- Take inspiration from existing branching strategies and create your own

### Integrating Changes & Structuring Releases

- The best approach depends on the needs and requirements of your team and project
- Option 1: Mainline development
    - New code is integrated quickly into the mainline, production code
    - few branches
    - relatively small commits
        - cannot risk big and bloated commits when constantly integrating into production code
    - high-quality testing and QA standards
- Option 2: State, Release, and Feature Branches
    - branches enhance structures and workflows
    - different types of branches fulfill different types of jobs
        - features and experiments are kept in different branches
        - releases can be planned and managed in their own different branches
        - different states in the development workflow can be represented by branches
    

### Types of Branches

- Long running
    - exist through the complete lifetime of the project
    - often, they mirror “stages” in your dev life cycle
    - common convention connected to long running branches
        - often no direct commits
        - commits are only added through merges or rebases
        - you don’t want to add untested code to production
        - might want to release new code in batches
- Short-lived
    - for new features, bug fixes, refactorings, experiments
    - will be deleted after integration (merge/rebase)
    - typically based on a long running branch

### Two Example Branching Strategies

- GitHub Flow
    - very simple, very learn
    - only one long running branch (”main”) + feature branches
- GitFlow
    - more structure, more rules
    - long-running: “main” + “develop”
        - main: a relfection of the current production state
        - develop: feature branches are based on it and will be merged back into it
            - also the starting point for any new releases
            - production ready versions are merged into main
    - short-lived: features, releases, hotfixes

## Pull Requests

- pull request are not a core git feature
- they are provided by your git hosting platform
- will work and look a bit different on different platforms
- without a pull request, you would jump right to merging your code
- they are a way to communicate about code and review it
- a way to contribute code to repositories you don’t have write access to
    - fork: your personal copy of a repository
        - You can make changes in your forked version and open a pull request to include those changes into the original
- pull requests are based on branches, not individual commits
- push branch to your remote fork
    - `git push --set-upstream <remote-branch> <local-branch>`
- request to include changes in original repository

## Merge Conflicts

- When they might occur
    - when integrating commits from different sources
        - git merge
        - git rebase
        - git pull
        - git stash apply
        - git cherry-pick
    - Git will mostly figure things out on its own
    - Can happen when contradictory changes happen
        - Git cannot decide which change it should keep
    - Git will immediately tell you when a conflict has occurred
    - Existing conflicts can be view with `git status`
- What they actually are
    - just characters in a file
    - Git marks the problematic areas in a file
- How to solve them
    - Undo a merge
        - `git merge --abort`
    - Undo a rebase
        - `git rebase --abort`
    - Clean up files that have been marked by Git

## Merge versus Rebase

- Merge
    - Git looks for three commits
        - The common ancestor commit
        - The last commit on branch A
        - The last  commit on branch B
    - Git creates a new commit that contains the differences between branch A and B
        - called a merge commit
        - automatically generated
        - would need to look at the commit history for both branches
- Rebase
    - not better or worse than merge, just different
    - makes history look like a straight line of commits without any branches
    - rewrites commit history
    - Steps
        - Git removes all commits on branch A and temporarily saves them
        - Git applies new commits from branch B
        - New commits from branch A are positioned on top of the commits from branch B
    - Do not rebase commits that you have already pushed to a shared repository
    - Use it for cleaning up your local commit history before merging it into a shared team branch




**References:**

* [Git for Professionals Tutorial - Tools & Concepts for Mastering Version Control with Git](https://www.youtube.com/watch?v=Uszj_k0DGsg)

