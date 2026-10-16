@bdd_cleaning
Feature: Cleaning story text
  A curator prepares a raw story before sentence marking.
  Cleaning removes a page-header line, flattens line breaks, simplifies quotes,
  and keeps letters, digits, spaces, and the characters [ " . ! ? |

  Rule: A page header is a line that starts with "Page | ", a number, and a space

    @smoke
    Scenario: A page header line is removed from the story
      Given a raw story:
        """
        Once upon a time.
        Page | 2 Harry Potter and the Philosophers Stone
        The end.
        """
      When the story is cleaned
      Then the prepared story is:
        """
        Once upon a time. The end.
        """

    Scenario: A story that is only a page header becomes empty
      Given a story that reads "Page | 9 Title here"
      When the story is cleaned
      Then the prepared story is empty

    Scenario: A page number without a trailing space stays in the story
      Given a story that reads "Page | 2"
      When the story is cleaned
      Then the prepared story reads "Page | 2"

    Scenario: A page header with only a trailing space is removed
      Given a page-number line that ends with a space
      When the story is cleaned
      Then the prepared story is empty

    Scenario: A line that is not a page header stays in the story
      Given a raw story:
        """
        Page | 2
        Page| 2 Title
        page | 2 Title
        """
      When the story is cleaned
      Then the prepared story reads "Page | 2 Page| 2 Title page | 2 Title"

  Rule: Cleaning simplifies quotes, spacing, and characters outside the story alphabet

    Scenario: Curly quotes become straight quotes and apostrophes are removed
      Given a raw story:
        """
        don’t “quote”
        """
      When the story is cleaned
      Then the prepared story is:
        """
        dont "quote"
        """

    Scenario: Characters outside the story alphabet become spaces
      Given a story that reads "Hello, world! Café #1."
      When the story is cleaned
      Then the prepared story reads "Hello world! Caf 1."

    Scenario: Hard line breaks become single spaces
      Given a raw story:
        """
        they were the last people
        youd expect
        """
      When the story is cleaned
      Then the prepared story reads "they were the last people youd expect"

    Scenario: Repeated spaces at the ends are collapsed
      Given a story that reads "  hello   world  "
      When the story is cleaned
      Then the prepared story reads "hello world"

    Scenario: A story of only spaces becomes empty
      Given a story of three spaces
      When the story is cleaned
      Then the prepared story is empty

  Rule: Cleaning leaves text alone when there is nothing to change or cleaning is off

    Scenario: Story cleaning turned off keeps the raw story
      Given a raw story:
        """
        Page | 1 Title
        hello
        """
      And story cleaning is turned off
      When the story is cleaned
      Then the story text is unchanged

    Scenario: An empty story stays empty
      Given an empty story
      When the story is cleaned
      Then the prepared story is empty

    Scenario: A story of plain words is unchanged
      Given a story that reads "hello world"
      When the story is cleaned
      Then the prepared story reads "hello world"
