---
aliases:
- /Notes-on-Git-Branches/
categories:
- git
- notes
date: '2021-12-29'
description: My notes from Tobias Gunther's video covering git branches.
hide: false
layout: post
search_exclude: false
title: Notes on Git Branches
toc: false

---


* [Overview](#overview)
* [The HEAD Branch](#the-head-branch)
* [Local and Remote Branches](#local-and-remote-branches)
* [Local and Remote Branches](#local-and-remote-branches)
* [Creating New Branches](#creating-new-branches)
* [List branches](#list-branches)
* [Switching Branches](#switching-branches)
* [Renaming Branches](#renaming-branches)
* [Publish Branch](#publish-branch)
* [Tracking Branches](#tracking-branches)
* [Pulling and Pushing Branches](#pulling-and-pushing-branches)
* [Deleting Branches](#deleting-branches)
* [Merging Branches](#merging-branches)
* [Rebasing Branches](#rebasing-branches)
* [Comparing Branches](#comparing-branches)



## Overview

Here are some notes I took while watching Tobias Gunther's [video](https://www.youtube.com/watch?v=e2IbNHi4uCI) covering git branches.



## The HEAD Branch

- The currently “active” or “checked out” branch
- only one can be active at a time

## Local and Remote Branches

- 99% of the time, “working” with branches means your local branches
- remote branches are more for synchronizing
    - GitHub
    - Git Lab
    - BitBucket
    - Azure DevOps

## Creating New Branches

- You can only create new branches in your local repository
- Create branches in a remote repository by publishing the branch in the local repository
- Based on your current HEAD branch
    - `git branch <new-branch-name>`
- Based on a different revision
    - `git branch <new-branch-name> <revision-hash>`

## List Branches

- `git branch`

## Switching Branches

- Current branch defines where new commits will be created
- Older
    - `git checkout <branch-name>`
    - Lots of different uses
- Newer
    - `git switch <branch-name>`
    - Specifically for switching branches

## Renaming Branches

- Rename local head branch
    - `git branch -m <new-name>`
- Rename different branch
    - `git branch -m <target-branch-name> <new-branch-name>`
- Rename remote branch
    1. Delete target branch
        - `git push origin --delete <old-name>`
    2. Publish new branch with desired name
        - `git push -u origin <new-name>`

## Publish Branch

- Upload a local branch for the first time
    - `git push -u origin <local-branch>`
- `-u` flag: Tells git to establish a tracking connection
    - makes pushing and pulling easier

## Tracking Branches

- Connecting branches with each other
- By default, local and remote branches have nothing to do with each other
- Get remote branch to local branch
    - `git branch --track <local-branch-name> <target-remote-branch>`
- Or:
    - `git checkout --track <target-remote-branch>`
    - Uses remote branch name as local branch name

## Pulling and Pushing Branches

- Synchronizing local and remote branches
- Much easier when tracking is already enabled
    - `git pull`
    - `git push`
- Git tells you if your local branch and tracked remote branch diverge
    - `git branch -v`

## Deleting Branches

- Cannot delete current head branch
    - Switch to other branch first
- Deleting a branch in your local repository
    - `git branch -d <branch-name>`
    - Might cause errors if you delete a branch with commits that do not exist elsewhere
        - `-f` flag: force deletion
        - Be careful with this option
- Deleting a remote branch
    - `git push origin --delete <remote-branch-name>`
- When deleting a branch, keep in mind whether you need to delete its remote/local counterpart as well

## Merging Branches

- Integrating changes from another branch into your current local HEAD branch
- Merging often produces a merge commit
- Switch to the branch that should receive changes
    - `git switch <branch-to-change>`
- Merge the branch with desired changes into current branch
    - `git merge <branch-with-changes>`

## Rebasing Branches

- An alternative way to integrate changes from another branch into your current local HEAD branch
    - Not really better or worse than merge, just different
    - There is no separate merge commit
    - It appears as if development history happened in a straight line
- Switch to the branch that should receive changes
    - `git switch <branch-to-change>`
- Rebase the branch with desired changes into current branch
    - `git rebase <branch with changes>`

## Comparing Branches

- Checking which commits are in branch-B, but not in branch-A
- Between two Local branches
    - `git log <branch-A>..<branch-B>`
- Between a Local and Remote branch
    - `git log <remote-branch>..<local-branch>`




**References:**

* [Git Branches Tutorial](https://www.youtube.com/watch?v=e2IbNHi4uCI)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->