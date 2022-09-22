---
title: Notes on Customizing Your GitHub Profile
date: 2021-12-9
image: /images/empty.gif
description: My notes from learning how to customize my GitHub profile.
categories: [github, notes]

---

* [Introduction](#introduction)
* [Basic Custom Profile](#basic-custom-profile)
* [Blog Post Feed](#blog-post-feed)
* [YouTube Channel Feed](#youtube-channel-feed)
* [Recent Activity Feed](#recent-activity-feed)
* [GitHub Stats](#github-stats)



## Introduction

I finally got around to making my [GitHub profile](https://github.com/cj-mills) look  less plain. I came across [this video](https://www.youtube.com/watch?v=ECuqb5Tv9qI) on YouTube covering several different customizations. Below are some notes I made for future reference.

### Basic Custom Profile

1. Create a repository that has the same name as your user name
2. Create a `README.md` file
    - Link to website
        - Use [https://shields.io](https://shields.io)
          
            [![Website](https://img.shields.io/website?label=christianjmills.com&style=for-the-badge&url=https://christianjmills.com)](https://christianjmills.com)
            
            ```markdown
            [![Website](https://img.shields.io/website?label=christianjmills.com&style=for-the-badge&url=https://christianjmills.com)](https://christianjmills.com)
            ```
        
    - Maybe Twitter
        - Use https://shields.io
          
            [![Twitter Follow](https://img.shields.io/twitter/follow/cdotjdotmills?color=1DA1F2&logo=twitter&style=for-the-badge)](https://twitter.com/intent/follow?original_referer=https://github.com/cj-mills&screen_name=cdotjdotmills)
            
            ```markdown
            [![Twitter Follow](https://img.shields.io/twitter/follow/cdotjdotmills?color=1DA1F2&logo=twitter&style=for-the-badge)](https://twitter.com/intent/follow?original_referer=https://github.com/cj-mills&screen_name=cdotjdotmills)
            ```
        
    - About Me section
    
    - Contacts info and social media
        - [<img align="left" alt="Channel Name | YouTube" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/youtube.svg" />](https://www.youtube.com)
        - `[<img align="left" alt="Channel Name | YouTube" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/youtube.svg" />][youtube_address]`
    - Languages and Tools that you work with
        - https://github.com/github/explore: Houses all of the community-curated content for GitHub Topics and Collections (e.g. icon images)
        
        - Copy Download link
        
        - Visual Studio
          
            [explore/visual-studio-code.png at main · github/explore](https://github.com/github/explore/blob/main/topics/visual-studio-code/visual-studio-code.png)
            
            - [<img align="left" alt="Visual Studio Code" width="26px" src="https://github.com/github/explore/raw/main/topics/visual-studio-code/visual-studio-code.png" />](https://code.visualstudio.com)
            - `[<img align="left" alt="Visual Studio Code" width="26px" src="https://github.com/github/explore/raw/main/topics/visual-studio-code/visual-studio-code.png" />](https://code.visualstudio.com)`
        
    - Can have separate definitions for links
        - `[website]: https://christianjmills.com`
3. Push the repository to GitHub.
4. Make sure the repository is public.

### Blog Post Feed

- Requires a link to an RSS `feed.xml` file

- Use GitHub Action
    - https://github.com/gautamkrishnar/blog-post-workflow
    
    - In `README.md`
      
        ```markdown
        # Blog posts
        <!-- BLOG-POST-LIST:START -->
        <!-- BLOG-POST-LIST:END -->
        ```
        
    - Create `.github` folder
        - Create `workflows` folder
            - Create `blog-post-workflow.yml` file
              
                ```yaml
                name: Latest blog post workflow
                on:
                  schedule: # Run workflow automatically
                    - cron: '0 * * * *' # Runs every hour, on the hour
                  workflow_dispatch: # Run workflow manually (without waiting for the cron to be called), through the Github Actions Workflow page directly
                
                jobs:
                  update-readme-with-blog:
                    name: Update this repo's README with latest blog posts
                    runs-on: ubuntu-latest
                    steps:
                      - name: Checkout
                        uses: actions/checkout@v2
                      - name: Pull in personal blog posts
                        uses: gautamkrishnar/blog-post-workflow@master
                        with:
                          feed_list: "https://christianjmills.com/feed.xml"
                ```
                
                - Additional options
                  
                    [GitHub - gautamkrishnar/blog-post-workflow: Show your latest blog posts from any sources or StackOverflow activity or Youtube Videos on your GitHub profile/project readme automatically using the RSS feed](https://github.com/gautamkrishnar/blog-post-workflow#options)
                    
                    - Include under jobs: → steps: → with:
        
    - Manually Update List
        - Go to `https://github.com/<user-name>/<repo-name>/actions/workflows/blog-post-workflow.yml`
        - Click on `Run workflow` drop-down menu
        - Click on `Run workflow` button

### YouTube Channel Feed

- Same steps as for Blog posts
- Changes
    - YouTube Channel Feed
        - `https://www.youtube.com/feeds/videos.xml?channel_id=<channel-id>`
    - `REAME.md`
      
        ```markdown
        # Youtube Videos
        <!-- YOUTUBE:START -->
        <!-- YOUTUBE:END -->
        ```
        
    - `.github/workflows/youtube-workflow.yml`
      
        ```yaml
        name: Latest YouTube video workflow
        on:
          schedule: # Run workflow automatically
            - cron: '0 * * * *' # Runs every hour, on the hour
          workflow_dispatch: # Run workflow manually (without waiting for the cron to be called), through the Github Actions Workflow page directly
        
        jobs:
          update-readme-with-blog:
            name: Update this repo's README with latest blog posts
            runs-on: ubuntu-latest
            steps:
              - name: Checkout
                uses: actions/checkout@v2
              - name: Pull in personal blog posts
                uses: gautamkrishnar/blog-post-workflow@master
                with:
        					comment_tag_name: "YOUTUBE"
                  feed_list: "https://www.youtube.com/feeds/videos.xml?channel_id=UCDOTuz8In9mVs44WZMbYNGg"
        ```
        

### Recent Activity Feed

- Use GitHub Action
    - https://github.com/jamesgeorge007/github-activity-readme
- Changes
    - `README.md`
      
        ```markdown
        # Recent GitHub Activity
        <!--START_SECTION:activity-->
        ```
        
    - `.github/workflows/update-readme.yml`
      
        ```yaml
        name: GitHub Activity
        
        on:
          schedule:
            - cron: '*/30 * * * *'
          workflow_dispatch:
        
        jobs:
          build:
            runs-on: ubuntu-latest
            name: Update this repo's README with recent activity
        
            steps:
              - uses: actions/checkout@v2
              - uses: jamesgeorge007/github-activity-readme@master
                env:
                  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        ```
        
    

### GitHub Stats

- Use GitHub Action
    - https://github.com/anuraghazra/github-readme-stats
    
- Add to `README.md`
    - Default
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>)
        ```
        
    - Hide Individual Stats
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>&hide=contribs,prs)
        ```
        
        - Options: `&hide=stars,commits,prs,issues,contribs`
        
    - Include Private Contributions
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>&count_private=true)
        ```
        
    - Show Icons
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>&show_icons=true)
        ```
        
    - Hide Border
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>&hide_border=true)
        ```
        
    - Themes
      
        ```markdown
        ![My GitHub stats](https://github-readme-stats.vercel.app/api?username=<user-name>&show_icons=true&theme=radical)
        ```
        
        - Built-in
          
            [github-readme-stats/README.md at master · anuraghazra/github-readme-stats](https://github.com/anuraghazra/github-readme-stats/blob/master/themes/README.md)
        
    - Customize
      
        [GitHub - anuraghazra/github-readme-stats: Dynamically generated stats for your github readmes](https://github.com/anuraghazra/github-readme-stats#customization)
        
    - Add Most Used Languages
      
        - ```markdown
            ![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=<user-name>&show_icons=true&hide_border=true)
            ```



 

**References:**

* [How To Create An Amazing Profile ReadMe With GitHub Actions](https://www.youtube.com/watch?v=ECuqb5Tv9qI)
