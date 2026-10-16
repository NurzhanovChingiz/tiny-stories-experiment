@bdd_dataset_preparation
Feature: Preparing a configured dataset
  A curator cleans each configured raw story file into the interim collection,
  then marks sentences in the processed collection. The raw file stays.
  Saving can be turned off. A missing file is recorded and is not invented.

  Rule: Cleaning and sentence marking write a new collection and leave the raw file

    @smoke
    Scenario: Cleaning a configured story file writes interim text
      Given a configured raw story file:
        """
        {"text": "Hello, world."}
        """
      When the curator cleans the dataset
      Then the interim story reads "Hello world."
      And the configured raw story file is unchanged
      And the curator is told the interim file was saved
      And the cleaning log records "Wrote cleaned file"

    @smoke
    Scenario: Normalizing a configured story file marks the sentence
      Given a configured cleaned story file:
        """
        {"text": "hello world."}
        """
      When the curator normalizes the dataset
      Then the processed story reads "hello world .|"
      And the curator is told the processed file was saved
      And the normalization log records "Wrote normalized file"

    Scenario: Cleaned stories land in the interim collection
      Given a configured raw story file:
        """
        {"text": "hello."}
        """
      When the curator cleans the dataset into the interim collection
      Then the interim story reads "hello."
      And the interim story is stored in the interim collection

    Scenario: Marked stories land in the processed collection
      Given a configured cleaned story file in the interim collection:
        """
        {"text": "hello world."}
        """
      When the curator normalizes the dataset into the processed collection
      Then the processed story reads "hello world .|"
      And the processed story is stored in the processed collection

  Rule: Saving can be turned off, and it cannot stay on while preparation is off

    Scenario: Saving cleaned stories turned off writes no interim file
      Given a configured raw story file:
        """
        {"text": "Hello."}
        """
      And saving cleaned stories is turned off
      When the curator cleans the dataset
      Then no interim story file is written
      And the cleaning log records "save_processed is false"

    Scenario: Saving marked stories turned off writes no processed file
      Given a configured cleaned story file:
        """
        {"text": "hello."}
        """
      And saving marked stories is turned off
      When the curator normalizes the dataset
      Then no processed story file is written
      And the normalization log records "save_processed is false"

    Scenario: Saving cleaned stories while cleaning is off is refused
      Given saving cleaned stories is turned on while cleaning is off
      When the curator cleans the dataset
      Then dataset preparation is refused because "save_processed requires enabled to be true"

    Scenario: Saving marked stories while sentence marking is off is refused
      Given saving marked stories is turned on while sentence marking is off
      When the curator normalizes the dataset
      Then dataset preparation is refused because "save_processed requires enabled to be true"

  Rule: A missing story file is recorded and not invented

    Scenario: A missing raw story file is not cleaned
      Given the configured raw story file is missing
      When the curator cleans the dataset
      Then dataset preparation is refused because "Raw story file does not exist"
      And the cleaning log records "Cleaning failed for valid.jsonl"

    Scenario: A missing cleaned story file is not normalized
      Given the configured cleaned story file is missing
      When the curator normalizes the dataset
      Then dataset preparation is refused because "Story file does not exist"
      And the normalization log records "Normalization failed for valid.jsonl"

  Rule: The curator can choose where the run is recorded

    Scenario: An explicit cleaning log receives the run record
      Given a configured raw story file:
        """
        {"text": "hello."}
        """
      And an explicit cleaning log
      When the curator cleans the dataset
      Then the explicit cleaning log records "INFO"
      And the default cleaning log is not created

    Scenario: An explicit normalization log receives the run record
      Given a configured cleaned story file:
        """
        {"text": "hello."}
        """
      And an explicit normalization log
      When the curator normalizes the dataset
      Then the explicit normalization log records "INFO"
      And the default normalization log is not created
