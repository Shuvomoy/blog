---
layout: post
title: Adding a GPG key manually in Ubuntu
categories: [programming]
comments: true 
---
Often in Ubuntu we need to execute lines like

    gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9

<!-- more --> Sometimes that may not work due to various reasons. There is a work around for that. 

First go to the site that hosts the key, which is in our case keyserver.ubuntu.com. In the search string box 
type: 

    0xE084DAB9

which is the key name in our case with 0x prefix. There may be several search results, select the one you are 
looking for. It will take you to a page starting with "Public Key Server&#x2026; ". Copy the contents of the page into a
text file. Save the text file. In terminal cd to the location that contains the text file, and type:

    sudo apt-key add <name of the text file >

That's it!