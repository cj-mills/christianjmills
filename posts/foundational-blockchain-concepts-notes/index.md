---
categories:
- web3
- notes
date: 2022-1-3
description: My notes from Lesson 0 of Patrick Collins' Solidity, Blockchain, and
  Smart Contract Course.
hide: false
layout: post
search_exclude: false
title: Notes on Foundational Blockchain Concepts
toc: false

aliases:
- /Notes-on-Foundational-Blockchain-Concepts/
---

* [Overview](#overview)
* [Bitcoin](#bitcoin)
* [Ethereum](#ethereum)
* [Smart Contracts](#smart-contracts)
* [Advantages Over Traditional Solutions](#advantages-over-traditional-solutions)
* [Getting an Ethereum Wallet](#getting-an-ethereum-wallet)
* [Make a Transaction on a Test Network](#make-a-transaction-on-a-test-network)
* [Block Explorers](#block-explorers)
* [Gas](#gas)
* [Blockchain Fundamentals](#blockchain-fundamentals)
* [Consensus](#consensus)
* [Scalability](#scalability)



## Overview

Here are some notes some notes I took while watching [Lesson 0](https://www.youtube.com/watch?v=M576WGiDBdQ&t=393s) of Patrick Collins' Solidity, Blockchain, and Smart Contract Course on freeCodeCamp.



## Bitcoin

- one of the first protocols to use blockchain technology
- took blockchain mainstream
- [Bitcoin Whitepaper](https://bitcoin.org/bitcoin.pdf)
  - Published by the Pseudonymous Identity, Satoshi Nakamoto in 2008
    - outlines how Bitcoin could be used to make peer-to-peer transactions on a decentralized network
- decentralized network powered by cryptography
- allows people to engage in censorship resistant finance in a decentralized manner
- Digital store of value
- Scarce and set amount that will ever be made

## Ethereum

- [Website](https://ethereum.org/en/)
- Ethereum Whitepaper
  
    [Ethereum Whitepaper | ethereum.org](https://ethereum.org/en/whitepaper/)
    
    - Published by Vitalik Buterin in 2013
- Uses blockchain infrastructure with an additional feature
- Added the capability for decentralized applications called smart contracts
- Ethereum is by far the most popular and most used smart contract protocol

## Smart Contracts

- Dapp = Smart Contract = Decentralized App
- Smart Contract: self executing set of instructions that is executed without a 3rd party intermediary
- Decentralized applications are usually a combination of several smart contracts
- Smart Contracts allow for agreements without centralized intermediaries
- The concept of smart contracts was originally introduced by [Nick Szabo](https://en.wikipedia.org/wiki/Nick_Szabo) in 1994
- Smart contracts are similar to traditional contracts, but is entirely written in code and automatically executed
- Bitcoin also has smart contracts, but they are not “Turing Complete”
    - this was intentional by the developers as they view the Bitcoin network as an asset
- The Oracle Problem
    - blockchains are deterministic systems
    - they are a walled garden
    - need a way to get real-world data
- Oracles
    - devices that bring data into a blockchain or execute some sort of external computation
    - centralized oracles are a point of failure
- Hybrid Smart Contracts
    - combining on-chain logic settlement layers with off-chain data and external computation
    - large majority of DeFi applications are hybrid smart contracts
- Chainlink
    - decentralized, modular oracle network
    - most popular and powerful
    - allows you to bring data into your smart contracts and do external computation
        - get data
        - get randomness
        - do some type of upkeep
    - allows for unlimited smart contract customization
    - blockhain and smart contract platform agnostic
        - it will work on basically any blockchain/smart contract platform

## Advantages Over Traditional Solutions

**Decentralized**

- there is no centralized entity that controls the blockchain
- node operators: independent individuals running the software that connects the blockchain
- enables users to live without a bank account
    - banks have the power to freeze  your funds

**Transparency and Flexibility**

- everything that is done on a blockchain and all the rules that are made can be seen by anybody
- there are no backdoor deals or special information
- everyone has to play by the same rules
- the blockchain is pseudo-anonymous

**Speed and Efficiency**

- since blockchains are verified by a decentralized collective, the settlement of transactions (e.g. making withdrawals) is substantially faster
- can take minutes or seconds depending on the blockchain used
- withdrawing from a bank can take multiple days
- stock trades can take up to a week to go through

**Security and Immutability**

- blockchains are immutable
    - they can’t be tampered with or corrupted
- a blockchain can handle many nodes going down without losing anything
- hacking a blockchain is substantially harder than hacking a centralized entity
- digital assets on a blockchain are more secure and mobile than physical assets

**Removal of Counterparty Risk**

- counterparty risk: the probability that the other party in an investment, credit, or trading transaction may not fulfill its part of the deal and may default on the contractual obligations
- help remove conflict of interests in agreements
    - Traditional Insurance:
        - You pay an insurance provider $100/month and in the event that you get hit by a bus, the insurance provider is supposed to pay you medical bills
        - The insurance provider has a massive conflict of interest holding up their end of the agreement
        - The insurance provider will look for loop holes to get out of paying
        - They are the ones who decide whether they will execute their end of the agreement
        

**Trust Minimized Agreements**

- smart contracts allow us to engage in trustless and trust minimized agreements
    - you don’t need to trust the other party to execute their end of an agreement
- Move from brand-based agreements to math-based agreements

**Decentralized Autonomous Organizations (DAOs)**

- organization that live online through smart contracts
- similar to a regular organization
- members hold governance tokens to make voting decisions
- all governance is done on chain

## [Getting an Ethereum Wallet](https://ethereum.org/en/wallets/find-wallet/)

- A wallet lets you connect to the Ethereum network and manage your funds
- Each wallet has a unique public account address
    - Can use tools like [Etherscan](https://etherscan.io/) to view different addresses and related transactions
    - A wallet can have multiple accounts
    - Each account has a public/private key pair for verifying transactions
        - Anyone can see the public key it and use it to verify that a transaction came from you and was signed using your private key.
        - The private key that is used to sign transactions and needs to be secret
            - signed transactions are verified by others using the public key
        - If you lose the private key, you lose access to the account
- Etheruem account address is derived from the your public key
    - hash the public key and take the last 20 bytes
- Private key is used to create the public key which is used to create the address
    - Private key → public key → address
- Wallets can switch between different networks
    - The primary Ethereum Network where tokens have actual value is called **Etheruem Mainet**
    - There are also test networks that are used when developing and testing applications before deploying them on the Mainnet
        - Currency on test networks do not have any real value
- Wallets have lots of optional features
    - Buying cryptocurrency directly with a bank card
    - Exploring dapps
    - Borrow, lend and earn interest directly from your wallet
    - Cash out of ETH and withdraw directly to a bank
    - Set limits that prevent your account from being drained
    - High-volume purchases
    - Trade between ETH and other tokens directly from your wallet
    - Multi-signature wallets require more than one account to authorize certain transactions
    

**[MetaMask](https://metamask.io/)**

- One of the most popular Etheruem wallets
- One of the easiest to use
- Installs as an [extension for your web browser](https://metamask.io/download.html) on desktop
    - Supports Chrome, Firefox, Brave, and Edge
- Can either create a new wallet or import an existing wallet using a seed phrase
    - New wallets will come with a secret backup phrase that can be used to restore an account. ***DO NOT LOSE OR SHARE THIS!!!***
- Use a secure password for the browser extension
- It is recommended to use a separate wallet for development purposes
- 

## Make a Transaction on a Test Network

[Faucets](https://faucets.chain.link/): A Test Network application that gives free test tokens that only work on a given test network

## Block Explorers

- An application that allows us to “view” transactions that happen on a blockchain.
- Transaction Details:
    - Transaction Hash
    - Status
    - Block containing the transaction record
    - Block confirmations: number of additional blocks added on after a transaction went through
    - Timestamp
    - Address that initiated the transaction
    - Smart Contract Address
    - Amount of tokens transferred
    - Value of tokens (zero for test networks)
    - Transaction Fee
    - Gas Price

## Gas

- Gas is a unit of computational measure. The more computation a transaction uses, the more “gas” you have to pay for
    - Sending ETH to 1 address would be cost less gas than sending ETH to 1,000 addresses
- Every transaction that happens on-chain pays a “gas fee” to node operators
    - Nodes can only put so many transactions into a block
    - Gas fees serve as incentive to add your transaction to a block
- Gas fees fluctuate depending on the amount of activity on the Ethereum network
    - The more people trying to make transactions, the higher the gas price, and therefore the higher the transaction fees
- Gas prices are set in [Gwei](http://eth-converter.com/)
    - `1` Gwei is equivalent to `0.000000001` ETH
- You can technically pick how much gas you want to pay
    - Can set gas limits to prevent transactions when gas fees are too high
    - Paying too small of a fee can significantly increase the transaction time or cause it to fail completely
    - You can pay more gas to have transaction processed more quickly
- Gas: Measure of computation use
- Gas Price: How much it costs per unit of gas
- Gas Limit: Max amount of gas in a transaction
- Transaction Fee: $Gas \ Used \cdot Gas \ Price$
- [ETH Gas Station](https://ethgasstation.info/): an online tool to estimate gas prices for different transaction priority levels

## Blockchain Fundamentals

**Hashes**

- A unique fixed length string, meant to identify a piece of data. They are created by placing said data into a “hash function”
- Hash Algorithm: a function that computes data into a unique hash
- Ethereum uses the [keccak256 hash algorithm](https://emn178.github.io/online-tools/keccak_256.html)
    - [Video](https://www.youtube.com/watch?v=wCD3fOlsGc4)

**Block**

- Data consists of a block number, nonce (a number), and  the piece of data
- nonce: a “number used once” to find the “solution” to problem used to verify the transactions in a block
    - also used to define the transaction number for an account/address
- All three pieces are fed into a has function to generate the hash value for a block
- Mining: the process of finding the “solution” to the blockchain problem used to verify the transactions in a block
- Nodes get paid for mining blocks
- Miners for a blockchain (when using Proof of Work) compute the hash values using different nonce values until they find the correct nonce value that generates a valid hash value
    - Miners might perform different tasks for different blockchains
- Block: a list of transactions mined together

**Blockchain**

- a sequence (chain) of blocks where each subsequent block keeps track of the hash value for the previous block
    - the hash value of the previous block is used. along with the current block number, nonce, and data to generate the hash value for the current block
    - altering a block earlier in the chain would thus require updating all subsequent blocks as well
    - this is how a blockchain can be considered immutable
- Genesis Block: the first block in a blockchain

**Distributed (Decentralized) Blockchain**

- multiple peers have a copy of the same blockchain
- anyone can become a peer on a blockchain
- a copy of a blockchain is considered valid if it matches the copies of a majority of the peers
    - each peer’s copy is weighted equally
- can quickly compare the copies of the blockchain by checking the last hash value in each copy as it is dependent on the hash values from all previous blocks in the chain

**Tokens**

- The data section in each block in a decentralized blockchain is replaced with record of transactions
- A transaction consists of a number of tokens (e.g. ETH) being sent from one address to another
- As mentioned earlier in the Wallets section, transactions are signed using the sender’s private key and verified using the sender’s public key

## Consensus

- the mechanism used to reach an agreement on the state of a blockchain
- Chain Selection:
    - Bitcoin and Ethereum use Nakamoto Consensus
        - combination of proof of work and longest chain rule
        - whichever chain that has the longest number of blocks on it is the one that will be used
        - 
- Sybil Resistance Mechanism: defines a way to figure out who is the block author
    - Which node is going to be the node that did the work
    - A blockchain’s ability to defend users creating a large number of pseudonymous accounts (i.e. a sybil attack) to gain a majority influence over the blockchain (i.e. a 51% attack)

**Proof of Work**

- a type of sybil resistance mechanism
- a node performs a computationally expensive task called mining
    - uses lots of energy
- Example: computing hash values
- block time: how long it takes between blocks being published
    - proportional to how hard the proof of work task is
- costs a lot of electricity
    - can lead to an environmental impact

**Proof of Stake**

- type of sybil resistance mechanism
- uses validators instead of miners
- proof of stake nodes put up collateral as a sybil resistance mechanism
    - if a node misbehaves, they lose some of their stake
- validators are chosen randomly
    - it is important to have verifiable randomness to ensure every validator has an equal chance of getting the reward
- Currently used by:
    - Avalanche
    - Solana
    - Polygon
    - Pokadot
    - Terra
    - Ethereum 2.0
- much more energy efficient than PoW
    - Only one node needs to do the work to figure out the new block and everyone else just validates it
- considered a slightly less decentralized solution as there is a minimum amount of tokens that a user needs to have to qualify for staking

## Scalability

- a situation where the cost to use a blockchain network increases the more people there are using it is not scalable
- Sharding: Ethereum 2.0’s solution to the scalability problem
    - a blockchain of blockchains
    - a main chain coordinates everything amongst several sub-chains
    - creates more chains for people to make transactions on, reducing cost per transaction

**Layer 1**

- base layer blockchain implementation
- Examples: Bitcoin, Ethereum

**Layer 2**

- any application that is added on top of a blockchain
- Example: Chainlink
- Can be used to solve the scalability problem using rollups
    - layer 2 transactions can be rolled up into a layer 1 transaction
- derive their security from a base layer 1 blockchain






**References:**

* [Solidity, Blockchain, and Smart Contract Course - Beginner to Expert Python Tutorial](https://www.youtube.com/watch?v=M576WGiDBdQ&t=393s)





<!-- Cloudflare Web Analytics --><script defer src='https://static.cloudflareinsights.com/beacon.min.js' data-cf-beacon='{"token": "56b8d2f624604c4891327b3c0d9f6703"}'></script><!-- End Cloudflare Web Analytics -->