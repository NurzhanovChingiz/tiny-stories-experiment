@bdd_normalization
Feature: Normalizing story text
  A curator marks sentence boundaries in a story that has already been cleaned.
  Normalization spaces sentence punctuation and inserts one "|" after each
  "." "!" or "?" token. It does not remove page headers.

  Rule: Sentence punctuation is spaced and marked once

    @smoke
    Scenario Outline: Each sentence punctuation mark is spaced and tagged
      Given a story that reads "Stop<mark>"
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "Stop <mark>|"

      Examples:
        | mark |
        | .    |
        | !    |
        | ?    |

    Scenario: Punctuation attached to words is separated and marked
      Given a story that reads "Hello.World!Fine?"
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "Hello .| World !| Fine ?|"

    Scenario: A sentence mark that is already spaced is tagged once
      Given a story that reads "hello world . They were fine"
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "hello world .| They were fine"

    Scenario: A story that already has the sentence mark is unchanged
      Given a story that reads "Hello world .|"
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "Hello world .|"

  Rule: Normalization formats tokens without cleaning the story

    Scenario: Quotes and brackets are separated from neighboring words
      Given a raw story:
        """
        She said"hello"[note].
        """
      When the story is normalized with the shipped sentence mark
      Then the prepared story is:
        """
        She said " hello " [ note] .|
        """

    Scenario: Repeated spaces are collapsed and the story is not marked
      Given a story that reads "  hello   world  "
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "hello world"

    Scenario: A page header stays in the story and its sentence is marked
      Given a story that reads "Page | 2 Harry Potter."
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "Page | 2 Harry Potter .|"

  Rule: Normalization leaves text alone when there is nothing to mark or it is off

    Scenario: Sentence marking turned off keeps attached punctuation
      Given a story that reads "hello world."
      And sentence marking is turned off
      When the story is normalized with the shipped sentence mark
      Then the story text is unchanged

    Scenario: An empty story stays empty
      Given an empty story
      When the story is normalized with the shipped sentence mark
      Then the prepared story is empty

    Scenario: A story of plain words is unchanged
      Given a story that reads "hello world"
      When the story is normalized with the shipped sentence mark
      Then the prepared story reads "hello world"
