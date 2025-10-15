---
title: "Notes on *The Superstruct Manifesto*"
date: 2025-02-09
image: /images/empty.gif
hide: false
search_exclude: false
categories: [professional-growth, book, notes]
description: "My notes from the book **The Superstruct Manifesto꞉ A Survival Guide for Founders Who Depend on Devs to Get Things Done** by **David Guttman**."

twitter-card:
 creator: "@cdotjdotmills"
 site: "@cdotjdotmills"
 image: /images/default-preview-image-black.png
open-graph:
 image: /images/default-preview-image-black.png
 
---







- [**Epigraph**](#epigraph)
- [**Introduction**](#introduction)  
- [**Chapter 1: We Will Not Inflict Daily Standups on Our Devs**](#chapter-1-we-will-not-inflict-daily-standups-on-our-devs)  
- [**Chapter 2: We Will Not Test Devs with Computer Science Riddles**](#chapter-2-we-will-not-test-devs-with-computer-science-riddles)  
- [**Chapter 3: We Will Not Recruit 10x Developers**](#chapter-3-we-will-not-recruit-10x-developers)  
- [**Chapter 4: We Will Not Let Devs Start without an Estimate**](#chapter-4-we-will-not-let-devs-start-without-an-estimate)  
- [**Chapter 5: We Will Not Sprint**](#chapter-5-we-will-not-sprint)  
- [**Chapter 6: We Will Not Allow Our Devs to Multitask**](#chapter-6-we-will-not-allow-our-devs-to-multitask)  
- [**Chapter 7: We Will Not Accept the First Solution a Dev Thinks Up**](#chapter-7-we-will-not-accept-the-first-solution-a-dev-thinks-up)  
- [**Chapter 8: We Will Not Allow Our Devs to Talk in Private**](#chapter-8-we-will-not-allow-our-devs-to-talk-in-private)  
- [**Chapter 9: We Will Not Allow Our Devs to Wander Off**](#chapter-9-we-will-not-allow-our-devs-to-wander-off)  
- [**Chapter 10: We Will Not Let Our Devs Boss Us Around**](#chapter-10-we-will-not-let-our-devs-boss-us-around)
- [**Conclusion**](#conclusion)  
- [**Key Takeaways**](#key-takeaways)





::: {.callout-note title="Book LInks:"}
* [Book Website](https://superstructmanifesto.com/)
:::



## **Epigraph**

> *"It's not because things are difficult that we dare not venture. It's because we dare not venture that they are difficult."*
> **– Seneca**



## **Introduction**

### **Foundational Points**

1. **Context**
   - Many founders fail for **avoidable reasons**.
   - There are **plenty of unpredictable problems** that can derail a startup; this guide focuses on **preventing the preventable**.

2. **Purpose of the Book**

   - **Survival guide** for founders.
   - The goal is to **avoid the worst and most common mistakes** when hiring and working with software engineers.

3. **Scope**
   - **Not a cookbook** with detailed recipes or step-by-step instructions.
   - Instead: an overview of **common pitfalls** and how to dodge them.

4. **Audience**

   - Written primarily for **smaller startups** or founders who **cannot afford** the same missteps as large, well-funded companies.
   - **Quote**: “If you are a big company or a unicorn startup that can afford to keep doing what you’re doing…This book is not for you.”

5. **Analogy**: Poisonous Plants and Pufferfish

   - In nature, many plants and mushrooms are deadly if not handled correctly.
   - Certain techniques might work at **large or specialized organizations** (e.g., pufferfish at fancy restaurants), but **founders shouldn’t blindly copy** those approaches for their own startups.

6. **Key Struggle**

   - Founders **struggle to hire and retain senior engineers**.

   - Many assume that getting senior engineers on board will solve all problems, but:

     - **Engineers can behave like seniors or juniors** regardless of technical skill level.
       - **Environment** created by the founder is what **encourages or discourages** the behaviors needed for success.

7. **Misconception about Experience**
   - Years of experience with a particular language or big-tech name brand does **not** guarantee the engineer can **solve the problems** your business actually faces.
   - Emphasis should be on **cultivating behaviors** and **structures** that support the company’s goals.

8. **Ownership and Structure**
   - If you **want engineers to behave like seniors**, you must **provide and enforce** the appropriate structure.
   - Some engineers may **push back** against imposed structure (they want “ice cream for dinner”), but **unfocused freedom** can lead to **low-value output** and **too many distractions**.



## **Chapter 1: We Will Not Inflict Daily Standups on Our Devs**

### **Overview**

1. **Position Statement**:
   - Daily stand-ups are **ubiquitous** yet considered **“stupid”** by the author.
   - They are seen as **expensive and disruptive** activities.
2. **Immediate Consequence**:
   - Daily stand-ups **guarantee devs never have a full day** of uninterrupted work.

### **Why Daily Stand-Ups Are Expensive**

1. **Time Not Spent Building**
   - Engineers are hired to **build** product features, fix bugs, and deliver value.
   - Any time spent in **stand-up meetings** is **time not spent building**.
2. **Cumulative Time Loss**
   - **Quote**: “If you have four devs standing around for 15 minutes every day, you are losing more than 20 hours of building time every month…”
   - That is **at least 20 hours** per month that **could have** produced a new feature or improvement.
3. **Worse Than It Looks**
   - Stand-ups are **supposed** to be 15 minutes, but the **context switching** before and after causes bigger disruptions.
   - Devs **can’t dive into** complex coding if they know they’ll be interrupted in a few minutes.
4. **Context Switching Costs**
   - Author’s analogy: “Software development is like **working at the bottom of the ocean**. Going up (interruptions) is time-consuming.”
   - Even if you reduce stand-up to 1 minute, there is **at least a 30-minute disruption** on either side.
5. **Total Hourly Loss**
   - For a 15-minute stand-up, the **absolute minimum** disruption is **over an hour per dev** (including the time lost to context switching and reorienting).
   - **Quote**: “If you stick to 15 minutes per meeting, you’re losing more than 100 hours a month of productive time with a team of four.”

### **Perceived Benefits vs. Actual Value**

1. **Potential Advantages**

   - Stand-ups supposedly:
     - Increase **productivity** and **transparency**.
     - Highlight **blockers**.
     - Improve **alignment** and **accountability**.

2. **Questionable Accountability**

   - **Reality**: Devs can “sound busy” by **exaggerating simple tasks**.
   - Many stand-up attendees **tune out** anyway.

3. **Transparency and Alignment**

   - The **ideal scenario**: Devs **learn** from each other’s updates, or catch potential **time-wasters** (e.g., a dev warns against a faulty OCR library).
   - This can happen, but the author compares it to **slot machines**—the “jackpot” scenario is **rare**.

4. **Alternatives**

   - You can achieve all the accountability and transparency stand-ups promise without daily interrupts:

     - **Asynchronous communication** to share status updates and blockers.

- Devs can **raise alerts** and help each other *when needed* rather than on a forced daily schedule.

### **Key Takeaway**

- **Quote**: “Nothing stands in the way of devs alerting their teammates to blockers.”
- Daily stand-ups are **inefficient** and can be **replaced** by **less disruptive** methods.



## **Chapter 2: We Will Not Test Devs with Computer Science Riddles**

### **Context**

1. **Author’s Warning**
   - Avoid **“Cracking the Coding Interview”** style questions for **startup hires**.
   - *Might* work for **Google, Amazon, Facebook**, but not necessarily for **most** startups.
2. **Real-World Example**
   - **Max Howell tweet**: Rejected by Google for failing to invert a binary tree on a whiteboard, despite writing **Homebrew**, used by 90% of Google’s own engineers.

### **Core Argument**

1. **Mismatch in Skills Needed**
   - If your **business** is not about **computer science riddles**, then these puzzle-based tests are **irrelevant**.
   - They **screen out** potentially great candidates who **already built** valuable software.
2. **What Startups Actually Need**
   - Typically, success = **building features** that **make customers happy**.
   - Deep algorithmic knowledge **may help** in niche cases, but the *main driver* is **customer-oriented value** creation.

### **Misguided Mimicry and Profiling Analogy**

1. **Caution Against Copying Big Tech**
   - A process that works for **Google or Facebook** (with **infinite candidates** and **massive budgets**) may **harm** a smaller startup’s hiring pipeline.
2. **Classification/Screening Problem**
   - Interviewing is like **detection or screening**. The **farther** you get from **testing what you actually care about**, the **worse** it is.
3. **Airport Security Analogy**
   - **Example**: In the wake of 9-11, some people argued we should focus attention on group A (Muslims) to detect behavior B (terrorism).
   - This approach introduces *huge complexities* and **may weaken** the system overall.
   - Similarly, focusing on whether candidates can do **binary tree inversion** vs. seeing if they can build product features is **misaligned**.
4. **Core Principle**
   - *“If you want to predict accurately whether a candidate will create value for your customers, you need to test for that.”*
   - Don’t rely on **off-the-shelf** coding riddles; **creatively test** for **actual tasks** that relate to **your** product or stack.

### **Practical Advice**

1. **Contractor Analogy**
   - You don’t ask a home contractor to solve logic puzzles. You **check their prior work** and **see if it’s relevant**.
2. **Testing Real Skills**
   - See if a candidate can **knock out real features** from your **actual roadmap** or from **similar** past experience.
3. **Focus on Fit**
   - People with heavy algorithmic training might **not** be the best fit for a small startup that prioritizes **direct customer impact**.



## **Chapter 3: We Will Not Recruit 10x Developers**

### **10x Developer Myth**

1. **Conventional Wisdom**
   - Many claim you should **only hire “10x developers.”**
   - Author says: This is **setting yourself up for failure**.
2. **Advice to Founders**
   - If an investor or executive **demands** you hire 10x devs, the author advises **ignoring** their further input on engineering.

### **Historical Origins**

1. **1960s Study**
   - The **“10x” concept** comes from research showing a few programmers were **10 times faster** at certain tasks than others in the group.
   - That study was done **decades ago** on a **12-programmer sample**, using a **massive, antiquated** mainframe.
   - Question: *How smart is it to rely on five-decade-old data?*
2. **Misinterpretation**
   - The 10x was between the **slowest** and the **fastest** in that specific test, **not** the fastest vs. the *average* developer.
   - The study’s environment is **vastly different** from modern-day development.

### **Practical Reality**

1. **Salary and Performance**
   - If a 10x dev truly existed, any **rational manager** would pay them far more to **keep them**.
   - This **breaks** the assumption that 10x devs are paid the same as everyone else and remain unnoticed.
2. **Situational Expertise**
   - The difference is often **task-based**. A dev with specialized experience can do a particular problem super fast, but might **not** excel at others.
   - Over a year, it’s **unlikely** to see a dev produce **10x** the output across **all** tasks.
3. **Author’s Own Example**
   - The author references building a **video editing app** in 10 hours. A PM estimated a team would need 200 hours.
   - This looks like the author is a **20x dev**, but the scenario is **cherry-picked**: it’s an app that fits the author’s personal expertise.

### **Commandos, Infantry, and Police Model**

1. **Commandos**
   - Highly creative problem-solvers who **establish a beachhead** quickly.
   - Great for **truly novel** or **R&D** tasks.
   - **Downside**: They get bored with routine tasks and may create drama by rewriting or overengineering.
2. **Infantry**
   - Solid, team-oriented devs who **execute proven designs** reliably.
   - They focus on **predictable delivery** and are typically **what most businesses need**.
3. **Police**
   - Specialists at **maintenance** tasks, security patches, bug fixes, and other routine work.
   - Commandos are **ill-suited** for these tasks; they’ll cause friction if forced into them.
4. **Hiring and Team Composition**
   - The author advises: Don’t fill your company with **too many** “commandos.”
   - Typically, **one** commando (or similarly skilled lead/consultant) is enough to handle novel problems or architecture decisions.
   - The **vast majority** of dev work is **parallelizable, routine, and incremental**—infantry and police excel here.

### **Conclusion on 10x Devs**

- **Focus** on building a **balanced team** that can consistently deliver.
- **Quote**: “Optimize for predictability, not the extremes.”



## **Chapter 4: We Will Not Let Devs Start without an Estimate**

### **Importance of Estimates**

1. **Controversy**
   - Software estimates are **notoriously** inaccurate.
   - Some companies abandon them entirely.
   - Author insists they are **still valuable**.

2. **Primary Reason**
   - Ownership and alignment:

     - For planning and coordinating with **other initiatives**.
   - **More importantly**, they get devs to **think through** the full project.

### **Estimates as a Management Tool**

1. **Immediate Improvement**
   - Simply **requiring** a dev to produce an estimate **improves** project quality.
   - Even if you **never look** at the estimate, it forces the dev to **consider the scope** before diving in.

2. **Analogy**
   - “Starting projects without an estimate is like **shopping without a budget**.”
   - Surprises of an **extra order of magnitude** in time or cost become visible **before** it’s too late.

3. **Scope Clarification**
   - Often, a big discrepancy between the dev’s estimate and your expectation reveals:

     - The dev may be **unsure** of the requirements.
       - The dev might **over-engineer** a piece that’s not crucial.
     - Or you missed some **hidden complexity**.


### **Constraints and Realities**

1. **Aligning Incentives**
   - Devs on salary or hourly rate have **no direct financial pressure** to ship quickly.
   - Overengineering and **scope creep** can thrive if you **don’t** push for estimates.
2. **Estimate Checkpoints**
   - Use the dev’s estimate to **check progress** at 25%, 50%, 75%, 90%.
   - If a dev is consistently behind schedule or **fails** to see red flags, it’s time to **intervene**.
3. **Handling Uncertainty**
   - **Quote**: “Measurements are observations that reduce uncertainty.”
   - Estimates don’t have to be **perfect**, just **close enough** (within ~15%) so that you can manage effectively.
4. **Accountability vs. Punishment**
   - **Don’t** punish devs for every single missed target.
   - Instead, watch for **consistency, communication, and conscientiousness**.
   - If a dev is **always** missing by 25% or more, they need **coaching** or there’s a deeper problem.
5. **Never Dictate an Estimate**
   - If you **demand** a specific timeline, devs feel no ownership.
   - If you disagree with the estimate, **reduce scope** or find a different dev.

### **Conclusion on Estimates**

- Estimates ensure **devs truly understand** the task and have **skin in the game**.
- They also help you **plan checkpoints** and **hold devs accountable**.



## **Chapter 5: We Will Not Sprint**

### **Two-Week Sprints Critique**

1. **Common Practice**
   - Sprints are **everywhere** in modern engineering.
   - The author calls them **“a bad idea.”**
2. **Caveat**
   - If the choice is between a **yearly release cycle** or two-week sprints, then sprints are obviously **better**.
   - But that **false dichotomy** doesn’t represent reality.

### **Sprints in Theory vs. Practice**

1. **Definition**
   - Typically a **two-week block** of time with assigned features/fixes.
   - At the end, results are **reviewed**.
2. **Alleged Benefits**
   - Forces tasks to be **broken down** into smaller deliverables.
   - Adds **urgency** and mitigates scope creep.
   - Improves **transparency** for stakeholders.
   - Encourages **retrospective learning**.
3. **Arbitrary Time Block**
   - The author calls the two-week schedule **“unnecessary”** in today’s environment.

### **Backpack Analogy**

1. **Fixed Capacity**
   - A sprint is like a **backpack** with limited space.
   - You try to fill it with an **optimal mix** of tasks.
   - You might end up with **unused** capacity or **overstuffing**.
2. **Guessing Game**
   - If you **underplan**, devs finish early and either become idle or break the sprint “rules.”
   - If you **overplan**, they inevitably **miss** the sprint deadlines.
3. **No Actual Benefit**
   - Stakeholders only care about **their feature** timeline, not an entire sprint’s success or failure.
4. **Better Alternatives**
   - You can still:
     - Break work into **small, shippable pieces**.
     - Defer **non-priority** tasks.
     - Meet regularly for **feedback**.
   - **None** of these require forcing everything into a **two-week cycle**.



## **Chapter 6: We Will Not Allow Our Devs to Multitask**

### **Value of Finished Projects**

1. **Zero Value Until Shipped**
   - If you have **10 partial projects**, the total value is still **zero**.
   - One **completed** project is better than many half-finished ones.
2. **Parallel vs. Serial Execution**
   - Example: 5 projects, each takes 2 days of dev work.
     - Doing them **in parallel** finishes **all on Day 10** → no benefits before Day 10.
     - Doing them **serially** yields **the first finished** by Day 2, second by Day 4, etc., which can **start delivering value** earlier.

### **Opportunity Costs and Profit Example**

1. **Simple Model**
   - 1 dev = $1/day cost.
   - 5 projects = each yields $1/day once finished, each requires 1 day of dev time.
   - Parallel approach = 0 revenue until end of Day 5 → net **-5** cost.
   - Serial approach = incremental revenue from Day 2 onward → net **+5** by Day 5.
2. **Immediate Feedback Loop**
   - Early completion of a project leads to **user feedback**, **analytics**, or **direct revenue** sooner.

### **Avoiding the Busy Trap**

1. **Dev Utilization**
   - Leaders often want devs “always busy,” but being busy with **multiple tasks** can delay actual **shipping**.
   - Dev waiting for code review should **focus on pushing the review along**, not start a new project and lose track of the old one.
2. **Ownership Through Deployment**
   - Devs shouldn’t say “not my problem” once they write the code.
   - True value is realized only when code is **live in production**.
3. **Context Switching Costs**
   - If a dev juggles multiple in-progress items, **re-familiarizing** themselves with older tasks after code review or QA feedback doubles the time.

### **Engineer Motivation**

1. **Blocking Issues**
   - Devs often claim they’re done if they’re waiting on someone else.
   - The author argues devs should remain **invested**: bug the reviewer, fix QA issues quickly, etc.
2. **Analogy**
   - Meeting a valuable contact for dinner: If you *really* care, you’ll find a way despite obstacles.
3. **Conclusion**
   - Don’t reward devs for simply **starting** multiple things. Reward them for **seeing things through** to production.



## **Chapter 7: We Will Not Accept the First Solution a Dev Thinks Up**

> “Go with your gut. Trust your instincts. Follow your intuition. They all have something in common. They're terrible strategies when you're making a decision.”

### **Why Instincts Can Be Misleading**

1. **Common Advice**
   - *“Go with your gut”* is **bad strategy** for engineering decisions.
   - Engineers’ **instincts** can be hijacked by **hidden biases**, personal preferences, or novelty-seeking.
2. **Typical Scenario**
   - You need a **simple** documentation site.
   - Engineer pushes for a fancy **framework** they’ve been reading about, leading to wasted time **over-engineering**.

### **Misaligned Priorities**

1. **Founders vs. Engineers**
   - Founders want **features** that **deliver value** quickly and reliably.
   - Engineers want **puzzles** and **fun new** tech to explore.
2. **When Quick-and-Dirty Is Best**
   - Sometimes, a single **HTML file** is enough if the site is only for a short-term purpose.
   - Overuse of React, Gatsby, or Next.js is **time-consuming** and might be **overkill**.
3. **Do Not Over-Correct**
   - Some devs always want the **slickest** approach. Others default to the **fastest hack**.
   - Problems emerge when they **cling** to a single approach **emotionally**.

### **How to Prevent Disaster**

1. **Founders Are the “What,” Engineers Are the “How.”**
   - If you **dictate** the technical approach, devs lose ownership and blame you if it fails.

2. **Require Multiple Approaches**
   - Don’t accept the first solution they come up with.

   - Ensure they provide at least three alternatives:

     - One focusing on **speed**,
     - One focusing on **low cost**,
     - One focusing on **robustness** or **full features**.

3. **Avoid Attachment**
   - When a dev presents only **one** solution, they **take it personally** if you reject it.
   - With **three solutions**, they can shift to alternative ideas without **ego battles**.

4. **Long-Term Benefit**

   - **Better** to take time up front, handle **uncomfortable** brainstorming, and find the approach that fits the **business constraints** (time, money, scope).



## **Chapter 8: We Will Not Allow Our Devs to Talk in Private**

### **Knowledge Retention**

1. **Risk of “In-the-Head” Data**
   - If crucial info is **only** in a key employee’s mind, it leaves with them if they quit.
   - A business reliant on a **single** person’s knowledge is **fragile**.
2. **Goal**
   - A valuable company can **thrive** independently of specific employees.
   - If the entire dev team vanished, could new devs **continue** from existing docs, repos, and infrastructure?
3. **RAM vs. Hard Drive** Analogy
   - Many companies store data **in engineers’ “RAM”** (their heads) instead of **persisting** it in a shared place.

### **Capturing Conversations**

1. **Recording and Searchability**
   - Imagine if **all** team discussions and decisions were **searchable** for future devs.
   - Onboarding becomes easier; leaving employees don’t take knowledge with them.
2. **Default to Text-Based Communication**
   - Instead of voice or video calls, use **chat or written** channels so everything is **automatically documented**.
3. **Resistance**
   - Some people find typing more **tedious** or **less fun**.
   - The author argues that’s a **worthwhile trade** for clarity, documentation, and consistency.
4. **Benefits of Writing**
   - Forces **more precise thought** and fosters a **lasting record**.
   - The company **should own** the knowledge, not let it vanish when employees depart.



## **Chapter 9: We Will Not Allow Our Devs to Wander Off**

### **Maintaining Motivation**

1. **Engineers Are Not Static**
   - Even great devs can **lose motivation** or **engagement** over time if **no leadership** is in place.
2. **Signs of Wandering**
   - Productivity slows; simple tasks take forever; more **careless** mistakes appear.
   - Dev might physically leave for another company or just **mentally “check out.”**
3. **One-on-One Meetings**
   - Frequent **personal check-ins** keep devs **centered** on business priorities.
   - Founder/manager must **directly address** the reasons devs stray.

### **Core Factors**

1. **Dev Must Want to Help You**
   - If dev feels you don’t **care about them**, they won’t **commit** 100%.
   - Basic **trust and empathy** go a long way.
2. **Dev Must Know How to Help You**
   - Many devs have a **distorted** view of the **company’s priorities**.
   - If you don’t clarify your **top goals**, devs can’t effectively align.
3. **Dev Must Avoid Blockers**
   - Dev complaints are **signals** of something that’s impeding them.
   - Could be **process issues**, missing docs, or dev’s own skill gaps.
   - Investigate systematically, rather than ignoring or always assuming dev error.
4. **Meaningful Work**
   - If devs **never** see the **impact** of their output, they lose motivation.
   - Founders know how features matter to customers, but devs may **not** see that connection.



## **Chapter 10: We Will Not Let Our Devs Boss Us Around**

### **Founder Fears**

1. **Devs Are Seen as Powerful**
   - They are expensive, hard to replace, and control the code.
   - Some founders worry about **displeasing** a lead dev who might quit, leaving the startup in trouble.
2. **Rewrite Demands**
   - The lead dev claims the **legacy code** is unacceptable, demands rewriting everything in a new framework, halting progress.
   - Founder feels they must **oblige** to avoid dev’s departure.
3. **Author’s Position**
   - This dynamic is **dangerous**. Founders must not feel **helpless**.
   - Devs **must** be held accountable and always keep **business objectives** in mind.

### **Business Realities vs. Dev Preferences**

1. **Customer Value Over Code Elegance**
   - Customers do not care if your **framework** is old or if the code is “legacy.”
   - They care about solutions that **save them time** or **money**.
2. **Tech Debt**
   - Tech debt **is real**, but a **complete rewrite** is an **extreme** measure.
   - Founders should ask devs **why** a rewrite is the only solution, how it affects **velocity**, and if it truly **reduces** user-facing bugs.
3. **Replaceability**
   - **Quote**: “Remember, your devs are not irreplaceable.”
   - A dev who **bullies** you into rewrites or halting your roadmap is a liability.
4. **Challenge Their Assumptions**
   - Make devs **prove** that a major refactor or rewrite is truly best for the **business** rather than an **engineering** preference.



## **Conclusion**

### **Overall Summary**

1. **Lessons from Others’ Mistakes**
   - The author has worked as a **consultant, CTO, and engineer** with various founders.
   - Common theme: Many mistakes could have been prevented by **sound leadership** and **recognizing engineering pitfalls** early.
2. **Key Points**
   - **Stand-ups** can be a costly **time sink**.
   - **Coding riddles** don’t reflect real job skills in many startup environments.
   - **10x developer** is a **myth** (or at least an overhyped concept).
   - **Software estimates** are crucial for **ownership**, not just planning.
   - **Sprints** are an **arbitrary** constraint, not a guaranteed productivity boost.
   - Devs should **finish** projects, not endlessly multitask.
   - Always **explore multiple solutions**; do not get attached to one approach.
   - Use **text-based communication** for knowledge persistence.
   - Keep devs **motivated** and **aligned** with company goals.
   - Don’t let devs **bully** you with demands that might not serve the **business**.

### **Adapting Advice**

1. **Don’t Mimic Big Tech Blindly**
   - The book cautions about **copying** well-known tech companies’ methods without considering **context**.
2. **Evidence-Based Practices**

   - If your **own experience** shows sprints (or stand-ups) work well, keep doing them.
   - The focus should be on **delivering value**, not **feeling** like a “real tech company.”
3. **Iterative Success**
   - Building a successful startup is an iterative process:

     1. Make decisions
     2. Reflect
     3. Adjust
4. **Courage and Honesty**
   - Retain what’s **effective** and drop what’s **not**, even if it’s an industry trend.



## **Key Takeaways**

- **Daily stand-ups**: Disruptive, often low-ROI, large **context-switching** cost.
- **CS riddle interviews**: Irrelevant to actual **customer-facing** work for most startups.
- **10x dev**: Usually a **myth**; hire **predictable, team-oriented** engineers.
- **Estimates**: Drive **accountability** and help detect scope issues **early**.
- **Avoid sprints** if they’re more about **ritual** than delivering real value.
- **Multitasking**: Finishing tasks one at a time often yields **faster** overall results.
- **Multiple approaches**: Encourage devs to brainstorm at least **three** solutions before deciding.
- **Document everything**: Default to **text-based** communication for knowledge retention.
- **One-on-ones**: Consistent check-ins keep devs aligned, motivated, and **unblocked**.
- **Do not fear your devs**: They work **for** you and **your customers’** benefit.

> **Above all**: Engineering strategies should serve the **business’s** success and **customer needs**.






{{< include /_about-author-cta.qmd >}}
