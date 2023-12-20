---
title: "Getting Started with Google Colab"
date: 2023-5-14
image: /images/empty.gif
hide: false
search_exclude: false
categories: [google-colab, getting-started, tutorial]
description: "Learn the fundamentals of Google Colab, a free cloud-based Jupyter Notebook environment, to write, run, and share Python code in your browser without any setup or installation."

twitter-card:
  creator: "@cdotjdotmills"
  site: "@cdotjdotmills"
  image: ../social-media/cover.jpg
open-graph:
  image: ../social-media/cover.jpg
---



* [Introduction](#introduction)
* [Access Google Colab](#access-google-colab)
* [The Notebook Selection Window](#the-notebook-selection-window)
* [Understanding the Notebook Interface](#understanding-the-notebook-interface)
* [Working with Data](#working-with-data)
* [Using Hardware Acceleration](#using-hardware-acceleration)
* [Create a New Notebook](#create-a-new-notebook)
* [Save Your Notebook](#save-your-notebook)
* [Conclusion](#conclusion)




## Introduction

In this tutorial, I'll introduce you to Google Colab, its features, and how to use it to run your code. Google Colab provides a free, cloud-based Jupyter Notebook environment that allows you to write, run, and share Python code in your browser without any setup or installation. A Jupyter Notebook is an interactive web-based tool for creating and sharing documents that contain live code, visualizations, and narrative text, often used in data analysis, visualization, and education.





## Access Google Colab

To access Google Colab, follow these steps:

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Sign in with your Google account. If you don't have a Google account, create one [here](https://accounts.google.com/signup).

![google-colab-welcome-notebook-signed-out](./images/google-colab-welcome-notebook-signed-out.png){fig-align="center"}



## The Notebook Selection Window

After signing in, the Notebook Selection window will pop up. This window displays a list of your recent notebooks, allowing you to access and open them. If this is your first time using Google Colab, you will only see the "Welcome to Colaboratory" notebook listed. The Notebook Selection window also allows you to import Jupyter Notebooks from Google Drive and GitHub or upload a notebook from your computer.



![google-colab-welcome-page](./images/google-colab-welcome-page.png){fig-align="center"}



The "Welcome to Colaboratory" notebook is already open behind the Notebook Selection window, so we'll work with that one. Click the cancel button in the bottom right corner of the popup window to view the welcome notebook.



![google-colab-welcome-page-exit-popup](./images/google-colab-welcome-page-exit-popup.png){fig-align="center"}









## Understanding the Notebook Interface

A notebook consists of a list of cells. Google Colab notebooks have two main types of cells: code cells and text cells. Code cells allow you to write and run Python code, while text cells let you add formatted text, images, and equations using [Markdown](https://www.markdownguide.org/getting-started/). The first few cells in the welcome notebook are text cells.



![google-colab-welcome-notebook-top](./images/google-colab-welcome-notebook-top.png){fig-align="center"}



### Text Cells

You can view and edit the Markdown source of a text cell by double-clicking it. In editor mode, Google Colab will show the Markdown source and the rendered version side-by-side.



![google-colab-welcome-notebook-inspect-markdown-source](./images/google-colab-welcome-notebook-inspect-markdown-source.png){fig-align="center"}





We can edit the Markdown source, and the rendered version will update in real time.



![google-colab-welcome-notebook-edit-markdown-source](./images/google-colab-welcome-notebook-edit-markdown-source.png){fig-align="center"}



You can exit the editor mode by pressing `Shift+Enter`, clicking the `Close Markdown Editor` icon in the top-right corner of the text cell, or clicking another cell.



![google-colab-welcome-notebook-exit-markdown-editor](./images/google-colab-welcome-notebook-exit-markdown-editor.png){fig-align="center"}



To create a new text cell, click the `+ Text` button in the toolbar. 



![google-colab-welcome-notebook-add-text-cell](./images/google-colab-welcome-notebook-add-text-cell.png){fig-align="center"}



Google Colab will add the new Markdown cell below the currently selected cell.



![google-colab-welcome-notebook-new-text-cell](./images/google-colab-welcome-notebook-new-text-cell.png){fig-align="center"}



### Code Cells
To create a new code cell, click the `+ Code` button in the toolbar at the top of the notebook or press `Ctrl+M B`. 

![google-colab-welcome-notebook-new-code-cell](./images/google-colab-welcome-notebook-add-new-code-cell.png){fig-align="center"}



Google Colab will add the new code cell below the currently selected cell. 



![google-colab-welcome-notebook-new-code-cell](./images/google-colab-welcome-notebook-new-code-cell.png){fig-align="center"}



You can write Python code in the code cell and execute it by pressing `Shift + Enter` or clicking the `Play` button on the left side of the cell. Any output from the code will appear directly below the code cell.



![google-colab-welcome-notebook-run-code-cell](./images/google-colab-welcome-notebook-run-code-cell.png){fig-align="center"}



We can also use code cells to access the command line by adding an exclamation point at the start of the cell. We can use this ability to install Python packages via the [pip](https://packaging.python.org/en/latest/key_projects/#pip) package installer.



![google-colab-welcome-notebook-access-command-line](./images/google-colab-welcome-notebook-access-command-line.png){fig-align="center"}







## Working with Data

Google Colab allows you to upload and download files to and from your computer and connect notebooks to your Google Drive.



### Uploading Files

You can upload files from your local machine to use in your Google Colab notebook by following these steps:

1. Click the Files icon in the left sidebar to open the file browser.

![google-colab-open-file-browser](./images/google-colab-open-file-browser.png){fig-align="center"}

2. Click the Upload button.

![google-colab-upload-button](./images/google-colab-upload-button.png){fig-align="center"}



3. Go to the file location on your local machine, select it, and click Open to upload it to your Google Colab workspace.

![google-colab-select-file-to-upload](./images/google-colab-select-file-to-upload.png){fig-align="center"}

4. Colab will display a warning that the runtime's files get deleted when it terminates. Click OK in the bottom-right corner of the popup window.

![google-colab-runtime-file-warning](./images/google-colab-runtime-file-warning.png){fig-align="center"}

5. The uploaded file will appear in the file browser, and you can access it in your notebook.

![google-colab-uploaded-file-in-file-browser](./images/google-colab-uploaded-file-in-file-browser.png){fig-align="center"}

6. We can view the file by double-clicking it in the file browser or loading it in the notebook via Python.

![google-colab-load-image-file](./images/google-colab-load-image-file.png){fig-align="center"}





### Downloading Files

To download a file from your Google Colab workspace to your local machine, follow these steps:

1. Locate the file in the file browser.
2. Right-click the file and select Download.

![google-colab-download-file](./images/google-colab-download-file.png){fig-align="center"}

The file will download to your local machine.







### Connecting to Google Drive

Google Colab notebooks can connect to Google Drive to access, store, and manage your files. To do this, follow these steps:

1. Click the Mount Drive button in the file browser.

![google-colab-mount-drive](./images/google-colab-mount-drive.png){fig-align="center"}

2. Google Colab will create a new code cell containing the following code:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

![google-colab-mount-drive-code-cell](./images/google-colab-mount-drive-code-cell.png){fig-align="center"}

3. Run the code cell by pressing `Shift + Enter` or clicking the Play button on the left side of the cell. A popup window will appear, prompting you to authorize access to your Google Drive.

![google-colab-mount-drive-code-cell-popup](./images/google-colab-mount-drive-code-cell-popup.png){fig-align="center"}

   

4. Click the `Connect to Google Drive` button to open the authorization page.

![google-colab-drive-authorization-page](./images/google-colab-drive-authorization-page.png){fig-align="center"}

5. Sign in with your Google account, and click Allow to grant access.

![google-colab-allow-google-drive-to-access-account](./images/google-colab-allow-google-drive-to-access-account.png){fig-align="center"}

6. Return to your Google Colab notebook. The code cell should have printed a message indicating your Google Drive is now mounted.

![google-colab-verify-google-drive-mounted-message](./images/google-colab-verify-google-drive-mounted-message.png){fig-align="center"}

7. Click the Refresh button in the file browser to update the contents.

![google-colab-refresh-file-browser](./images/google-colab-refresh-file-browser.png){fig-align="center"}

Your Google Drive should now be mounted and accessible from the file browser.

![google-colab-verify-driver-accessible-in-file-browser](./images/google-colab-verify-driver-accessible-in-file-browser.png){fig-align="center"}

You can read, write, and manage your Google Drive files directly from your Google Colab notebook. To access the files, use the path `/content/drive/MyDrive/` followed by the file or folder name.







## Using Hardware Acceleration

Google Colab offers free access to GPUs and TPUs to accelerate your code. To enable GPU or TPU acceleration:

1. Click the "Runtime" menu at the top of the notebook.

![google-colab-click-runtime-menu](./images/google-colab-click-runtime-menu.png){fig-align="center"}

2. Select "Change runtime type."

![google-colab-runtime-menu-change-runtime-type](./images/google-colab-runtime-menu-change-runtime-type.png){fig-align="center"}

3. Choose "GPU" from the "Hardware accelerator" drop-down menu.

![google-colab-choose-gpu-hardware-accelerator](./images/google-colab-choose-gpu-hardware-accelerator.png){fig-align="center"}

4. Click "Save."

![google-colab-save-hardware-accelerator-selection](./images/google-colab-save-hardware-accelerator-selection.png){fig-align="center"}

5. Changing the hardware accelerator requires loading a new runtime. Loading a new runtime will delete any files we added and disconnect Google Drive. Google Colab will show a popup window asking you to confirm you want to delete the current runtime. Click "OK" to confirm.

![google-colab-delete-previous-runtime-popup-window](./images/google-colab-delete-previous-runtime-popup-window.png){fig-align="center"}

6. Verify the notebook has GPU access by running the following code in a code cell:

   ```bash
   !nvidia-smi
   ```

![google-colab-nvidia-smi-results](./images/google-colab-nvidia-smi-results.png){fig-align="center"}



Your notebook will now use the selected hardware accelerator. Note that free GPU and TPU usage is time-limited. You can run notebooks on the free tier for at most 12 hours at a time (usually less). If you exceed the time allotment, you must wait until it resets (typically about 12 hours). Therefore, only enable hardware acceleration when needed and disable it when you don't. To disable hardware acceleration, select None from the Hardware Accelerator drop-down menu.

![google-colab-disable-hardware-acceleration](./images/google-colab-disable-hardware-acceleration.png){fig-align="center"}






## Create a New Notebook

To create a new notebook:

1. Open the `File` menu in the top-left corner and select `New notebook`.

![google-colab-create-new-notebook](./images/google-colab-create-new-notebook.png){fig-align="center"}



A new notebook will open in a separate tab. The runtime for the previous notebook is still active.

![google-colab-new-notebook](./images/google-colab-new-notebook.png){fig-align="center"}



You can rename the notebook by clicking the notebook name at the top of the page. For now, we can name it "My First Notebook."

![google-colab-rename-notebook](./images/google-colab-rename-notebook.png){fig-align="center"}







## Save Your Notebook

Google Colab automatically saves your notebooks to a "Colab Notebooks" folder in Google Drive.

![google-drive-colab-notebooks-folder](./images/google-drive-colab-notebooks-folder.png){fig-align="center"}



Note the "Welcome to Colaboratory" notebook is not in the folder. Since we did not create that notebook, we must save our copy manually. Switch to that notebook's tab and click the "Copy to Drive" button.

![google-colab-save-copy-to-drive](./images/google-colab-save-copy-to-drive.png){fig-align="center"}



Google Colab will open our new copy of the notebook in a separate tab.

![google-colab-copy-of-welcome-notebook](./images/google-colab-copy-of-welcome-notebook.png){fig-align="center"}




If we check the "Colab Notebooks" folder in Google Drive, we should now see our copy of the welcome notebook.

![google-drive-verify-copy-of-welcome-notebook](./images/google-drive-verify-copy-of-welcome-notebook.png){fig-align="center"}







## Sharing Notebooks

You can share your Google Colab notebook with others, similar to other Google Drive documents. To share your notebook:

1. Click the "Share" button in the top-right corner of the notebook.

![google-colab-share-button](./images/google-colab-share-button.png){fig-align="center"}

2. In the sharing settings dialog box, enter the email address of the person you want to share the notebook with. 

![google-colab-share-notebook-add-people](./images/google-colab-share-notebook-add-people.png){fig-align="center"}

3. Select their permission level: "Viewer," "Commenter," or "Editor."

![google-colab-share-notebook-set-permission-level](./images/google-colab-share-notebook-set-permission-level.png){fig-align="center"}

   

   Alternatively, you can create a shareable link with specific access settings (view, comment, or edit). You can share this link with others, and they can access the notebook according to the chosen permission level.

   1. Open the drop-down menu under General Access and select the "Anyone with a link" option.

   ![google-colab-enable-shareable-link](./images/google-colab-enable-shareable-link.png){fig-align="center"}

   2. Set the permission level: "Viewer," "Commenter," or "Editor."

   ![google-colab-enable-shareable-link-permission-level](./images/google-colab-enable-shareable-link-permission-level.png){fig-align="center"}

   3. Click "Copy link" to copy the shareable link.

   ![google-colab-enable-shareable-link-copy-link](./images/google-colab-enable-shareable-link-copy-link.png){fig-align="center"}



## Version Control with GitHub

Google Colab can save and load notebooks from GitHub repositories, enabling seamless collaboration and tracking of changes in your code.



### Saving a notebook to a GitHub repository:

1. Open the File menu in the top-left corner and select Save a copy in GitHub.

![google-colab-file-menu-save-copy-in-github](./images/google-colab-file-menu-save-copy-in-github.png){fig-align="center"}

2. If you haven't connected your GitHub account yet, follow the prompts to authorize Google Colab to access your repositories.

![github-authorize-google-colab](./images/github-authorize-google-colab.png){fig-align="center"}

3. Choose a repository, branch, and file path for your notebook. You can also update the commit message.

![google-colab-copy-to-github-select-repository](./images/google-colab-copy-to-github-select-repository.png){fig-align="center"}

4. Click OK to save the notebook to the specified GitHub repository.

![google-colab-copy-to-github-click-ok](./images/google-colab-copy-to-github-click-ok.png){fig-align="center"}



### Loading a notebook from a GitHub repository:

1. Go to the [Google Colab website](https://colab.research.google.com/).

![google-colab-welcome-page](./images/google-colab-welcome-page.png){fig-align="center"}{fig-align="center"}

2. Click the GitHub tab in the Notebook Selection window.

![google-colab-selection-window-click-github-tab](./images/google-colab-selection-window-click-github-tab.png){fig-align="center"}

3. Enter the URL for the GitHub repository containing the notebook you want to open. You can also search for one by entering a username or organization and repository name.

![google-colab-enter-github-repo-url](./images/google-colab-enter-github-repo-url.png){fig-align="center"}

4. Select the notebook you want to open, and it will open in a new tab.

   



## Conclusion

You've now learned the fundamentals of Google Colab. This tutorial covered creating and editing cells, working with data, hardware acceleration, and saving and sharing notebooks via Google Drive and GitHub.
Keep exploring Google Colab to uncover more features that can enhance your projects.





