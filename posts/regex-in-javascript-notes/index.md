---
aliases:
- /javascript/notes/regex/2021/12/29/Notes-on-Regex-in-JavaScript
categories:
- javascript
- regex
- notes
date: '2021-12-29'
description: My notes from freeCodeCamp's video covering regular expressions in JavaScript.
hide: false
layout: post
search_exclude: false
title: Notes on Regular Expressions in JavaScript
toc: false

---

* [Overview](#overview)
* [Regular Expressions](#regular-expressions)
* [Using the Test Method](#using-the-test-method)
* [Match Literal Strings](#match-literal-strings)
* [Match a Literal String with Different Possibilities](#match-a-literal-string-with-different-possibilities)
* [Ignore Case While Matching](#ignore-case-while-matching)
* [Extract Matches](#extract-matches)
* [Find More Than the First Match](#find-more-than-the-first-match)
* [Match Anything with Wildcard Period](#match-anything-with-wildcard-period)
* [Match Single Character with Multiple Possibilities](#match-single-character-with-multiple-possibilities)
* [Match Letters of the Alphabet](#match-letters-of-the-alphabet)
* [Match Numbers and Letters of the Alphabet](#match-numbers-and-letters-of-the-alphabet)
* [Match Single Characters Not Specified](#match-single-characters-not-specified)
* [Match Characters that Occur One or More Times](#match-characters-that-occur-one-or-more-times)
* [Match Characters that Occur Zero or More Times](#match-characters-that-occur-zero-or-more-times)
* [Find Characters with Lazy Matching](#find-characters-with-lazy-matching)
* [Find One or More Criminals in a Hunt](#find-one-or-more-criminals-in-a-hunt)
* [Match Beginning String Patterns](#match-beginning-string-patterns)
* [Match Ending String Patterns](#match-ending-string-patterns)
* [Match All Letters and Numbers](#match-all-letters-and-numbers)
* [Match Everything But Letters and Numbers](#match-everything-but-letters-and-numbers)
* [Match All Numbers](#match-all-numbers)
* [Match All Non-Numbers](#match-all-non-numbers)
* [Restrict Possible Usernames](#restrict-possible-usernames)
* [Match Whitespace](#match-whitespace)
* [Match Non-Whitespace Characters](#match-non-whitespace-characters)
* [Specify Upper and Lower Number of Matches](#specify-upper-and-lower-number-of-matches)
* [Specify Only the Lower Number of Matches](#specify-only-the-lower-number-of-matches)
* [Check for All or None](#check-for-all-or-none)
* [Positive and Negative Lookahead](#positive-and-negative-lookahead)
* [Reuse Patterns Using Capture Groups](#reuse-patterns-using-capture-groups)
* [Use Capture Groups to Search and Replace](#use-capture-groups-to-search-and-replace)
* [Replace Whitespace from Start and End](#replace-whitespace-from-start-and-end)



## Overview

Here are some notes I took while watching freeCodeCamp's [video](https://www.youtube.com/watch?v=ZfQFUJhPqMM) covering covering regular expressions for JavaScript.

Online JavaScript Console

* [Replit - JavaScript Console](https://replit.com/@innominate817/JavaScriptConsole#index.js)



## Regular Expressions

- Define a search pattern that can be used to search for things in a string



## Using the Test Method

- takes a regex, applies it to a string and returns True or False based on whether the pattern matches something

```jsx
let sentence = "The dog chased the cat."
// Do not need quote marks in the regex
let regex = /the/

let myString = "Hello, World!";
let myRegex = /Hello/;
// Use test method
// should return true
let result = myRegex.test(myString);
console.log(result);
```

## Match Literal Strings

```jsx
let waldoIsHiding = "Somewhere Waldo is hiding in this text.";
// Case sensitive
let waldoRegex = /Waldo/;
let result = waldoRegex.test(waldoIsHiding);
console.log(result);
```

## Match a Literal String with Different Possibilities

```jsx
let petString = "James has a pet cat.";
// Look for the words dog, cat, bird, or fish
let petRegex = /dog|cat|bird|fish/;
// Should return true as cat is present
let result = petRegex.test(petString);
console.log(result);
```

## Ignore Case While Matching

```jsx
let myString = "freeCodeCamp";
// Ignores case when looking for matches
let fccRegex = /freecodecamp/i;
let result = fccRegex.test(myString);
console.log(result);
```

## Extract Matches

```jsx
let extractStr = "Extract the word 'coding' from this string.";
let codingRegex = /coding/;
// Extracts the substring matching the provided regex
let result = extractStr.match(codingRegex);

console.log(result);
```

## Find More Than the First Match

```jsx
let testStr = "Repeat, Repeat, Repeat";
// Adding the g flag will find every occurrence
let ourRegex = /Repeat/g;
let result1 = testStr.match(ourRegex);
console.log(result1);

let twinkleStar = "Twinkle, twinkle, little star";
// Will ignore case ***and*** find every occurrence
let starRegex = /twinkle/ig;
let result2 = twinkleStar.match(starRegex);

console.log(result2);
```

## Match Anything with Wildcard Period

```jsx
let humStr = "I'll hum a song";
let hugStr = "Bear hug";
let huRegex = /hu./;
// Matches any words that start with "hu"
console.log(humStr.match(huRegex));
// Matches any words that start with "hu"
console.log(hugStr.match(huRegex));

let exampleStr = "Let's have run with regular expressions!";
// Match any word ending with "un"
let unRegex = /.un/;
let result = unRegex.test(exampleStr);
console.log(result);
```

## Match Single Character with Multiple Possibilities

```jsx
// Matches "bag", "big", "bug"
let bgRegex = /b[aiu]g/;

let quoteSample = "Beware of bugs in the code; I have only proved it corr";
// Matches every vowel and ignores case
let vowelRegex = /[aeiou]/ig;
let result = quoteSample.match(vowelRegex);

console.log(result);
```

## Match Letters of the Alphabet

```jsx
let quoteSample = "The quick brown fox jumped over the lazy dog.";
let alphabetRegex = /[a-z]/ig;
let result = quoteSample.match(alphabetRegex);

console.log(result);
```

## Match Numbers and Letters of the Alphabet

```jsx
let quotsample = "Blueberry 3.141592653s are delicious.";

// Match any digits in range [1-9]
// Also match any letters "a" through "s" and ignore case 
let myRegex = /[a-s|2-6]/ig;
console.log(quotsample.match(myRegex));
```

## Match Single Characters Not Specified

```jsx
let quoteSample = "3 blind mice.";
// Match any characters not containing
// numbers or vowels
let myRegex = /[^0-9aeiou]/ig;
console.log(quoteSample.match(myRegex));
```

## Match Characters that Occur One or More Times

```jsx
let difficultSpelling = "Mississippi";
// Match any occurence of the letter "s" one or more times
let myRegex = /s+/g;
console.log(difficultSpelling.match(myRegex));
```

## Match Characters that Occur Zero or More Times

```jsx
let soccerWord = "gooooooooal!";
let gPhrase = "gut feeling";
let oPhrase = "over the moon";
let goRegex = /go*/;
// Returns ["goooooooo"]
console.log(soccerWord.match(goRegex));
// Returns ["g"]
console.log(gPhrase.match(goRegex));
// Retrurns null
console.log(oPhrase.match(goRegex));

let chewieQuote = "Aaaaaaaaaaaaaaaarrrgh!";
// Match "a" zero or one times
let chewieRegex = /Aa*/;
console.log(chewieQuote.match(chewieRegex));
```

## Find Characters with Lazy Matching

* greedy match: finds the longest possible part of a string that fits the regex pattern
  - regex patterns default to greedy


* lazy match: finds the smallest part of the string that fits the regex pattern

```jsx
let string = "titanic";
// Looks for words that start with "t" followed by zero or more letters and ends with "i"
// Adding a "?" makes it a lazy match 
let regex = /t[a-z]*?i/;
console.log(string.match(regex));

let text = "<h1>Winter is coming</h1>";
// Looks for a substring starting with "<" followed zero or more characters and ends with ">"
// Lazy match
let myRegex = /<.*?>/;
console.log(text.match(myRegex));
```

## Find One or More Criminals in a Hunt

```jsx
let crowd = 'P1P2P3P4P5P6PCCCP7P8P8';
// Matches one or more instances of "C"
let reCriminals = /C+/;
console.log(crowd.match(reCriminals));
```

## Match Beginning String Patterns

```jsx
let rickyAndCal = "Cal and Ricky both like racing.";
// Match string that starts with "Cal"
let calRegex = /^Cal/;
console.log(calRegex.test(rickyAndCal));
```

## Match Ending String Patterns

```jsx
let caboose = "The last car on a train is the caboose";
// Match string that ends with "caboose"
let lastRegex = /caboose$/;
console.log(lastRegex.test(caboose));
```

## Match All Letters and Numbers

```jsx
let quoteSample = "The five boxing wizards jump quickly";
// \w: Shorthand for matching all letters, case insensitive, all digits, and underscore
let alphabetRegexV2 = /\w/g;
let result = quoteSample.match(alphabetRegexV2);
console.log(result);
console.log(result.length)
```

## Match Everything But Letters and Numbers

```jsx
let quoteSample = "The five boxing wizards jump quickly.";
// \W: Shorthand for matching everything that is not a letter, digit, or underscore
let nonAlphatbetRegex = /\W/g;
let result = quoteSample.match(nonAlphatbetRegex);
console.log(result);
console.log(result.length)
```

## Match All Numbers

```jsx
let numString = "Your sandwich will be $5.00";
// \d: shorthand for matching all digits
let numRegex = /\d/g;
let result = numString.match(numRegex);
console.log(result);
console.log(result.length);
```

## Match All Non-Numbers

```jsx
let numString = "Your sandwich will be $5.00";
// \D: shorthand for matching all non digits
let noNumRegex = /\D/g;
let result = numString.match(noNumRegex);
console.log(result);
console.log(result.length);
```

## Restrict Possible Usernames

```jsx
/*
1) If there are numbers, they must be at the end.
2) Letters can be lowercase and uppercase.
3) At least two characters long. Two-letter names can't have numbers.
*/

let username = "JackOfAllTrades2";
// {2,}: number of times the previous pattern can match (two or more)
let userCheck = /^[A-za-z]{2,}\d*$/;
console.log(userCheck.test(username));
```

## Match Whitespace

```jsx
let sample = "Whitespace is important in separating words";
// Match all whitespace characters
let countWhiteSpace = /\s/g;
console.log(sample.match(countWhiteSpace));
```

## Match Non-Whitespace Characters

```jsx
let sample = "Whitespace is important in separating words";
// Match all non-whitespace characters
let countNonWhiteSpace = /\S/g;
console.log(sample.match(countNonWhiteSpace));
```

## Specify Upper and Lower Number of Matches

```jsx
let ohStr = "Ohhh no";
// Only match "Ohhh no" through "Ohhhhhh no"
let ohRegex = /Oh{3,6} no/;
console.log(ohRegex.test(ohStr));
```

## Specify Only the Lower Number of Matches

```jsx
let haStr = "Hazzzzah";
// Need to have at least four "z" after "Ha"
let haRegex = /Haz{4,}/;
console.log(haRegex.test(haStr));
```

Specify Exact Number of Matches

```jsx
let timStr = "Timmmmber";
// Match exactly four "m" between "Ti" and "ber"
let timRegex = /Tim{4}ber/;
console.log(timRegex.test(timStr));
```

## Check for All or None

```jsx
let favWord = "favorite";
// Check for all or none instances of "u"
let favRegex = /favou?rite/;
console.log(favRegex.test(favWord));
```

## Positive and Negative Lookahead

* lookaheads: patterns that tell JavaScript to look ahead in your string to check for patterns further along. Can be useful when checking for multiple patterns over the same string. 

```jsx
let quit = "qu"
let noquit = "qt"
// Positive lookahead
// Looks for "q" makes sure "u" occurs later in the string 
let quRegex = /q(?=u)/;
// Negative lookahead
// Looks for "q" makes sure "u" does not occur later in the string
let qRegex = /q(?!u)/;
// Returns ["q"];
console.log(quit.match(quRegex));
// Returns ["q"]
console.log(noquit.match(qRegex));

let sampleWord = "astro22naut";
// Match words that are greater than five characters and have two consecutive
let pwRegex = /(?=\w{5})(?=\D*\d{2})/;
console.log(pwRegex.test(sampleWord));
```

## Reuse Patterns Using Capture Groups

```jsx
let repeatStr = "regex regex";
// Equivalent to /(\w+)\s(\w+)/
let repeatRegex = /(\w+)\s\1/;
// Returns true
console.log(repeatRegex.test(repeatStr));
// Returns ["regex regex", "regex"]
console.log(repeatStr.match(repeatRegex));

let repeatNum = "42 42 42";
// Equivalent to /(\d+)\s(\d+)\s(\d+)\s/
let reRegex = /^(\d+)\s\1\s\1$/;
console.log(reRegex.test(repeatNum));
console.log(repeatNum.match(reRegex));
```

## Use Capture Groups to Search and Replace

```jsx
let wrongText = "The sky is silver.";
let silverRegex = /silver/;
// Returns "The sky is blue."
console.log(wrongText.replace(silverRegex, "blue"));

// Returns "Camp Code"
// Matches two groups of characters separated by a whitespace
console.log("Code Camp".replace(/(\w+)\s(\w+)/, '$2 $1'));

let huhText = "This sandwich is good.";
// Find the word "good"
let fixRegex = /good/;
// Replace it with "okey-dokey" 
let replaceText = "okey-dokey";
console.log(huhText.replace(fixRegex, replaceText));
```

## Replace Whitespace from Start and End

```jsx
let hello = "    Hello, World!  ";
let wsRegex = /^\s+|\s+$/g;
console.log(hello.replace(wsRegex, ""))
```



**References:**

* [Learn Regular Expressions (Regex) - Crash Course for Beginners](https://www.youtube.com/watch?v=ZfQFUJhPqMM)
