@bdd_acquisition
Feature: Acquiring a published dataset
  A curator copies the configured story files into the raw collection.
  A file whose bytes already match the publication is left in place.
  A failed copy does not invent a raw file or replace one that is already there.

  Rule: A missing raw file is copied from the publication

    @smoke
    Scenario: A published story file is saved into the raw collection
      Given a published story file "notes.txt" that reads "notes"
      When the curator acquires the dataset
      Then the curator is told "notes.txt" was saved
      And the raw file "notes.txt" reads "notes"
      And the publication was copied
      And the acquisition log records "Stored raw file"

  Rule: A matching raw file is left in place unless the curator asks to replace it

    Scenario: A matching story file is left in place
      Given a published story file "notes.txt" that reads "one"
      And the raw collection already holds a matching "notes.txt"
      When the curator acquires the dataset
      Then the curator is told "notes.txt" is already present
      And the raw file "notes.txt" reads "one"
      And the publication was not copied
      And the acquisition log records "wrote 0 file(s), kept 1 file(s)"

    Scenario: Replacing copies the published story file again
      Given the curator replaces story files that already match
      And a published story file "notes.txt" that reads "new"
      And the raw collection already holds "notes.txt" reading "old"
      When the curator acquires the dataset
      Then the curator is told "notes.txt" was saved
      And the raw file "notes.txt" reads "new"
      And the publication was copied

    Scenario: An empty raw file is copied again
      Given a published story file "notes.txt" that reads "notes"
      And the raw collection already holds an empty "notes.txt"
      When the curator acquires the dataset
      Then the curator is told "notes.txt" was saved
      And the raw file "notes.txt" reads "notes"
      And the publication was copied

  Rule: Acquisition is refused when the request or the copy cannot be stored

    Scenario: A missing dataset configuration is refused
      Given no dataset configuration
      When the curator acquires the dataset
      Then acquisition is refused because "Dataset configuration file does not exist"
      And no raw story file is stored

    Scenario: A publication that cannot be copied stores nothing
      Given a published story file "notes.txt" that cannot be copied
      When the curator acquires the dataset
      Then acquisition is refused because "Could not download notes.txt"
      And no raw story file is stored
      And the acquisition log records "Acquisition failed for notes.txt"

    Scenario: A later copy failure leaves the earlier story file
      Given a published story file "train.jsonl" that reads "train"
      And a published story file "valid.jsonl" that cannot be copied
      When the curator acquires the dataset
      Then acquisition is refused because "Could not download valid.jsonl"
      And the raw file "train.jsonl" reads "train"
      And the raw file "valid.jsonl" is absent

    Scenario: A mismatched published copy leaves the original story file
      Given the curator replaces story files that already match
      And the raw collection already holds "notes.txt" reading "old"
      And a published story file "notes.txt" whose copy does not match its digest
      When the curator acquires the dataset
      Then acquisition is refused because "Could not download notes.txt"
      And the raw file "notes.txt" reads "old"

    Scenario: A file name that leaves the raw collection is refused
      Given the dataset lists the file "../escape.jsonl"
      When the curator acquires the dataset
      Then acquisition is refused because "single file name"
      And no raw story file is stored

    Scenario: An unreachable publication keeps a recorded story file
      Given a published story file "notes.txt" that reads "notes"
      And that story file was recorded from an earlier acquisition
      And the publication cannot be reached
      When the curator acquires the dataset
      Then the curator is told "notes.txt" is already present
      And the raw file "notes.txt" reads "notes"
      And the publication was not copied

    Scenario: An unreachable publication with nothing recorded is refused
      Given a published story file "notes.txt" that reads "notes"
      And the publication cannot be reached
      When the curator acquires the dataset
      Then acquisition is refused because "Could not download notes.txt"
      And no raw story file is stored
