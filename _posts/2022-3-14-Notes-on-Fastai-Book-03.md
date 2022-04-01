---
title: Notes on fastai Book Ch. 3
layout: post
toc: false
comments: true
description: My full notes for chapter 3 of Deep Learning for Coders with fastai & PyTorch
categories: [ai, ethics, fastai, notes]
hide: false
permalink: /:title/
search_exclude: false
---



* [Data Ethics](#data-ethics)
* [Key Examples for Data Ethics](#key-examples-for-data-ethics)
* [Integrating Machine Learning with Product Design](#integrating-machine-learning-with-product-design)
* [Topics in Data Ethics](#topics-in-data-ethics)
* [Identifying and Addressing Ethical Issues](#identifying-and-addressing-ethical-issues)
* [Role of Policy](#role-of-policy)
* [References](#references)



## Data Ethics


- Sometimes machine learning models can go wrong
    - the can have bugs
    - they can be presented with data that they have not seen before and behave in ways we don’t expect
    - they can work exactly as designed but be used for malicious purposes
- the power of deep learning makes it important for us to consider the consequences of our choices

### Ethics

- the philosophical study of right and wrong, including how we can define those terms, reognize right and wrong actions, and understand the connection between actions and consequences
- no one really agrees on  what right and wrong are, whether they exist, how to spot them, which people are good and bad, etc.
- [What is Ethics?](https://www.scu.edu/ethics/ethics-resources/ethical-decision-making/what-is-ethics/) by Markkula Center for Applied Ethics
    - the term ethics refers to
        - Well-founded standards of right and wrong that prescribe what humans should do
        - The study and development of one’s ethical standards
- ethics is complicated and context-dependent
- involves the perspectives of many stakeholders

### What is Data Ethics?

- a subfiled of ethics

- being used to define policy in many jurisdictions

- being used in companies  to consider how best to ensure good societal outcomes from product development

- being used by researchers who want to make sure the work they are doing is used for good

- deep learning practitioners will likely face situations where they need to consider data ethics

  

## Key Examples for Data Ethics

### Bugs and Recourse: Buggy Algorithm Used for Healthcare Benefits

- Akansas’s buggy healthcare system left patients stranded
- [What happens when an algorithm cuts your health care](https://www.theverge.com/2018/3/21/17144260/healthcare-medicaid-algorithm-arkansas-cerebral-palsy)
    - The Verge investigated software used in over half of the US states to determine how much healthcare people receive
    - hundreds of people, many with severe disabilities, had their healthcare drastically cut.
        - a woman with cerebral-palsy how needs constant assistance had her hours of help suddenly reduced by 20 hours per week
            - she could not get any explanation for why
            - a court case revealed there were mistakes made in the software implementation of the algorithm

### Feedback Loops: YouTube’s Recommendation System

- feedback loops can occur when your model is influences future input data
- YouTube’s recommendation system helped unleash a conspiracy theory boom
    - The algorithm was designed to optimize watch time
    - responsible for 70% of the content that is watched
- [YouTube Unleashed a Conspiracy Theory Boom. Can It Be Contained?](https://www.nytimes.com/2019/02/19/technology/youtube-conspiracy-stars.html)
- recommendation systems have a lot of power over what people see

### Bias: Professor Layanya Sweeney “Arrested”

- When a traditionally African American name is searched for on Google, is displays ads for criminal background checks
- Dr. Latanya Sweeney is a professor at Harvard and director of the university’s data privacy lab
    - discovered that Googling her name resulted in adverstisements saying “Latanya Sweeney Arrested?”, despite being the only Latanya Sweeney and never having been arrested
    - discovered that historically Black names received advertisements suggesting the person had a criminal record
    

### Why Does This Matter?

- everybody who is training models needs to consider how their models will be used and consider how to ensure they are used positively
    - How would you feel if you discovered that you had been part of system that ended up hurting society?
        - Would you be open to finding out?
        - How can you help make sure this doesn’t happen?
- IBM and Nazi Germany
    - [IBM and the Holocaust](https://www.dialogpress.com/books/ibm-and-holocaust) by Edwin Black
        - “To the blind technocrat, the means were more important than the ends. The destruction of the Jewish people became even less important because the invigorating nature of IBM’s technical achievement was only heightened by the fantastical profits to be made at a time when bread lines stretched across the world.”
    - IBM supplied the Nazis with the data tabulation products used to track the extermination of Jews and other groups on a massive scale
        - This was drive from the top of the company, with marketing to Hitler and his leadership team
        - Company President Thomas Watson personally approved the 1939 release of special IBM alphabetizing machines to help organize the deportation of Polish Jews
        - Hitler awared Company President Thomas Watson with a special “Service to the Reich” medal in 1937
        - IBM and its subsidiaries provied regular training and maintenance onsite at the concentration camps
            - printing off cards
            - configuring machines
            - repariing machines
        - IBM set up categorizations on its punch card system for
            - the way each person was killed
            - whic group they were assigned to
            - the logistical information necessary to track them through the vast Holocaust system
- Volkswagen Diesel Scandal
    - the care company was revealed to have cheated on its diesel emissions tests
    - the first person who was jailed was one of the engineers who just did what he was told
- Data scientists need to consider the entire pipeline of steps that occurs between the development of a model and the point at which the model is used to make a decision
    - inform everyone involved in this chain about the capabilities, constraints, and details of your work
    - make sure the right issues are being considered
- need to know when to refuse to do a piece of work



## Integrating Machine Learning with Product Design

- Lots of decisions are involved when collecting your data and developing your model
    - What level of aggregation will you store your data at?
    - What loss function should you use?
    - What validation and training sets should you use?
    - Should you focus on simplicity of implementation, speed of inference, or accuracy of the model?
    - How will your model handle out-of-domain data items?
    - Can it be fine-tuned, or must it be retrained from scratch over time?
- Whoever ends up developing and using the system that implements your model will not be well-placed to understand the decisions you made when training the model
- Data scientists need to be part of a tightly integrated, cross-disciplinary team
- Researchers need to work closely with the kinds of people who will end up using their research.
- Ideally, domain experts should learn enought to be able to train and debug some models themselves



## Topics in Data Ethics

### Recourse and Accountability

- a lack of responsibility and accountability leads to bad results
- mechanisms for data audits and error corrections are crucial
    - data often contains errors
- machine learning practitioners have a responsibility to understand how their algorithms end up being implemented in practice

### Feedback Loops

- be aware of the centrality of metrics in driving  a fincancially important system
    - an algorithm will do everything it can to optimize a target metric
        - can lead to unforeseen edge cases
        - humans ineracting with a system will search for, find, and exploit these edge cases and feedback loops for their advantage
    - transparancy is important for uncovering problems
- there can also be feedback loops without humans
- consider what data features and metrics might lead to feedback loops
    - the most optimal algorithm might not be the best one to deploy to production
- have mechanisms in place to spot runaway feedback loops and tak positive steps to break them when they occur

### Bias

- [A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle](https://arxiv.org/abs/1901.10002)
- Six Types of Bias in Machine Learning
    1. Historical Bias
        - bias from people, processes, and society
        - historical bias is a fundamental, structural issue with the first step of the data generation process
        - can exist even given perfect sampling and feature selection
        - any dataset involving humans can have historical bias
            - medical data
            - sales data
            - housing data
            - political data
        - fixing problems in machine learning systems is hard when the input data has problems
        - under representation of certain groups in the training data can result in a model that performs worse for those subgroups
            - [No Classification without Representation: Assessing Geodiversity Issues in Open Data Sets for the Developing World](https://arxiv.org/abs/1711.08536)
                - the vast majority of images in popular benchmark datasets like ImageNet were found to be from the US and other Western countries
            - [Does Object Recognition Work for Everyone?](https://arxiv.org/abs/1906.02659)
                - object recognition models trained on popular benckmark datasets performed poorly on items from lower-income countries
    2. Representation Bias
        - [Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting](https://arxiv.org/abs/1901.09451)
        - models can not only reflect representation imbalances, but also amplify them
    3. Measurement Bias
        - bias from measuring the wrong thing, measuring something in the wrong way, or incorporating a measurement into the model innapropriately
        - [Does Machine Learning Automate Moral Hazard and Error?](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf)
    4. Aggregation Bias
        - occurs when models do not aggregate data in a way that incorporates all of the appropriate factors, or when a model does not include the necessary interactions terms, nonlinearities, or so forth
        - likely to occur in medical settings
    5. Evaluation Bias
    6. Deployment Bias

### Addressing different types of bias

- different types of bias require different approaches for mitigations
- Create better documentation
    - decisions
    - context
    - specifics about how and why a particular dataset was created
    - what scenerios it is appropriate to use in
    - what the limitations are
- Algorithms are used differently than human decision makers
    - in practice, machine learning is often implemented because it is cheaper and more efficient, not because it leads to better outcomes
    - [Weapons of Math Destruction](https://www.penguinrandomhouse.com/books/241363/weapons-of-math-destruction-by-cathy-oneil/)
        - describes a pattern in which the priviledged are processed by people, whereas the poor are processed by algorithms
    - People are more likely to assume algorithms are objective or error-free
    - Algorithms are more likely to be implemented with no appeals process in place
    - Algorithms are often used at scale
    - Algorithmic systems are often cheap

### Disinformation

- often used to sow disharmony and uncertainty, and to get people to give up on seeking the truth
    - receiving conflicting accounts can lead people to assume they can’t trust anyting
- often contains seeds of truth, or half truths taken out of context
- Most propaganda campaigns are a carefully designed mixture of facts, half-truths, exagerrations, and deliberate lies.
    - [A Houston protest, organized by Russian trolls](https://www.houstonchronicle.com/local/gray-matters/article/A-Houston-protest-organized-by-Russian-trolls-12625481.php)
- often involves coordinated campaigns of inauthentic behavior
    - fraufulent accounts may try to make it seem like many people hold a particular viewpoint
- disinformation through autogenerated texsst is a significant issue enabled by deep learning models
- one proposed solution is to develope some form of digital signiture
- [How Will We Prevent AI-Based Forgery?](https://hbr.org/2019/03/how-will-we-prevent-ai-based-forgery)
    - “AI is poised to make high-fidelity forgery inexpensive and automated, leading to potentially disastrous consequences for democracy, security, and society.”



## Identifying and Addressing Ethical Issues

- make finding and dealing with mistakes part of the design of any system that includes machine learning

### Analyze a Project You Are Working On

- Should we even be doing this?
- What bias is in the data?
- Can the code be audited?
- What are the error rates for different subgroups?
- What is the accuracy of a simple rule-based alternative?
- What processes are in place to handle appeals or mistakes?
- How diverse is the team that built it?
- What data are you collecting and storing?
    - data often ends up being used for different purposes than the original intent
        - census data has been used to target minorities
        - [How Capitalism Betrayed Privacy](https://www.nytimes.com/2019/04/10/opinion/sunday/privacy-capitalism.html)

### Implement processes at your company to find and address ethical risks

- [An Ethical Toolkit for Engineering/Design Practice](https://www.scu.edu/ethics-in-technology-practice/ethical-toolkit/)
    - includes concrete practices to implement at your company
        - regularly scheduled sweeps to proactively search for ethical risks
        - expanding the ethical circle to include the perspectives of a variety of stakeholders
        - considering how bad actors can abuse, steal, misinterpret, hack, destroy, or weaponize what your are building
    - Whose interests, desires, skills, experiences, and values have we simply assumed, rather than actually consulted?
    - Who are all the stakeholders who will be directly affected by our product? How have their interests been protected? How do we know what their interests really are?
    - Who/which groups and individuals will be indirectly  affected in significant ways?
    - Who might use this product the we did not expect to use it, or for purposes we did not initially intend?

**Ethical lenses**

- different foundational ethical lenses can help identify concrete issues
- The rights approach
    - Which option best respects the rights of all who have a stake?
- The justics approach
    - Which option treats people equally or proportionately?
- The utilitarian approach
    - Which option will produce the most good and do the least harm?
- The common good approach
    - Which option best serves the community as a whole, not just some members?
- The virtue approach
    - Which option leads me to act as the sort of person I want to be?
- Consquences
    - Who will be directly affected bt this project? Who will be indirectly affected?
    - Will the effects in aggregate likely create more good than harm, and what types of good and harm?
    - Are we thinking about all relevant types of harm/benefit?
    - How might future generations be affected by this project?
    - Do the risks of harm from this project fall disproportionately on the least powerful? Will the benefits go disproportionately go to the well-off?
    - Have we adequately considered “dual-use” and unintended downstream effects?
- Deontological perspective
    - What rights of other and duties to others must we respect?
    - How might the dignity and autonomy of each stakeholder be impacted by this project?
    - What considerations of trust and of justice are relevant to this design/project?
    - Does this project involve any conflicting moral duties to other, or conflicting stakeholder rights? Howe can we prioritize these?

### The Power of Diversity

- team members from similar backgrounds are likely to have similar blindspots around ethical risks
- diversity can lead to problems being identified earlie and a wider range of solutions being considered

### Fairness, Accountability, and Transparency

- [Fairness and machine learning](https://fairmlbook.org/) book
    - gives a perspective on machine learning that treats fairness as a central concern rather than an afterthought
- Exercise
    1. Come up with a process, definition, set of questions, etc. which is designed to resolve a problem
    2. Try to come up with an example in which that apparant solution results in a proposal that no one would consider acceptable
    3. This can then lead to further refinement of the solution



## Role of Policy

- purely technical solutions are not sufficient to address the underlying problems that have led to our current state.
    - as long as it is profitable to create addictive technology, companies will continue to do so

### The Effectiveness of Regulation

- corporations are more reactive to the threat of significant financial penalty than to the systematic destruction of an ethnic minority

### Rights and Policy

- many harms resulting from unintended consequences of misuses of technology involve public goods, such as a polluted information environment or deteriorated ambient privacy
- there are societal impacts to widespread survilance
- human rights issues need to be addressed through law

### Cars: A Historical Precedent

- The movement to increase car safety sets a historical precedent for addressing problems in data ethics
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [The Nut Behind the Wheel](https://99percentinvisible.org/episode/nut-behind-wheel/)




## References

* [Deep Learning for Coders with fastai & PyTorch](https://www.oreilly.com/library/view/deep-learning-for/9781492045519/)
* [The fastai book GitHub Repository](https://github.com/fastai/fastbook)