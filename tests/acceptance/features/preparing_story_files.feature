@bdd_story_files
Feature: Preparing story files
  A curator rewrites each story's text into a new JSONL file.
  Other fields stay, the raw file stays, blank lines are skipped, and a failed
  rewrite does not replace a file that is already there.

  Rule: A successful rewrite changes story text and leaves the raw file

    @smoke
    Scenario: Cleaning a story file simplifies text and keeps the other fields
      Given a raw story file:
        """
        {"text": "Hello, world.", "id": 3}
        """
      When the story file is cleaned
      Then the derived story file is:
        """
        {"text": "Hello world.", "id": 3}
        """
      And the raw story file is unchanged

    @smoke
    Scenario: Normalizing a story file marks the sentence and keeps the other fields
      Given a raw story file:
        """
        {"text": "hello world.", "story_id": 7}
        """
      When the story file is normalized
      Then the derived story file is:
        """
        {"text": "hello world .|", "story_id": 7}
        """
      And the raw story file is unchanged

    Scenario: Blank lines are not treated as stories
      Given a raw story file:
        """
        {"text": "Hello."}

        {"text": "World."}
        """
      When the story file is cleaned
      Then the derived story file is:
        """
        {"text": "Hello."}
        {"text": "World."}
        """

    Scenario: Blank lines are not treated as stories when sentences are marked
      Given a raw story file:
        """
        {"text": "Hello."}

        {"text": "World."}
        """
      When the story file is normalized
      Then the derived story file is:
        """
        {"text": "Hello .|"}
        {"text": "World .|"}
        """

  Rule: A rewrite is refused when the file is missing, unsafe, or not a story

    Scenario: A missing raw file is not cleaned
      Given no raw story file
      When the story file is cleaned
      Then preparation is refused because "Raw story file does not exist"
      And no derived story file is written

    Scenario: A missing story file is not normalized
      Given no raw story file
      When the story file is normalized
      Then preparation is refused because "Story file does not exist"
      And no derived story file is written

    Scenario: Cleaning refuses to replace the raw story file
      Given a raw story file:
        """
        {"text": "hello."}
        """
      And the derived file is the raw story file
      When the story file is cleaned
      Then preparation is refused because "Refusing to replace raw story file"
      And the raw story file is unchanged

    Scenario: Normalizing refuses to replace the story file
      Given a raw story file:
        """
        {"text": "hello."}
        """
      And the derived file is the raw story file
      When the story file is normalized
      Then preparation is refused because "Refusing to replace story file"
      And the raw story file is unchanged

    Scenario: A line that is not a story object leaves the existing derived file
      Given a raw story file:
        """
        {"text": "ok"}
        []
        """
      And an existing derived story file:
        """
        KEEP
        """
      When the story file is cleaned
      Then preparation is refused because "JSONL line 2 must be an object"
      And the derived story file is unchanged

    Scenario: A story without text is not normalized
      Given a raw story file:
        """
        {"id": 1}
        """
      When the story file is normalized
      Then preparation is refused because "JSONL line 1 must contain a string text field"
      And no derived story file is written

    Scenario: A non-text story field is not cleaned
      Given a raw story file:
        """
        {"text": 1}
        """
      When the story file is cleaned
      Then preparation is refused because "JSONL line 1 must contain a string text field"
      And no derived story file is written

    Scenario: Invalid JSON does not create a derived file
      Given a raw story file:
        """
        not json
        """
      When the story file is cleaned
      Then preparation is refused because "Could not clean"
      And no derived story file is written

    Scenario: Invalid JSON does not create a normalized file
      Given a raw story file:
        """
        not json
        """
      When the story file is normalized
      Then preparation is refused because "Could not normalize"
      And no derived story file is written

    Scenario: A line that is not a story object is not normalized
      Given a raw story file:
        """
        {"text": "ok"}
        []
        """
      And an existing derived story file:
        """
        KEEP
        """
      When the story file is normalized
      Then preparation is refused because "JSONL line 2 must be an object"
      And the derived story file is unchanged

    Scenario: A non-text story field is not normalized
      Given a raw story file:
        """
        {"text": 1}
        """
      When the story file is normalized
      Then preparation is refused because "JSONL line 1 must contain a string text field"
      And no derived story file is written
