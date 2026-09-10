from debug_assistant.datasets.patch_parser import parse_unified_patch


def test_parse_patch_separates_edit_and_hunk_ranges():
    patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -10,4 +10,4 @@
 context
-old
+new
 context
 context
"""
    result = parse_unified_patch(patch)["files"][0]
    assert result["old_edit_ranges"] == [{"start_line": 11, "end_line": 11}]
    assert result["new_edit_ranges"] == [{"start_line": 11, "end_line": 11}]
    assert result["hunk_ranges"][0] == {
        "old": {"start_line": 10, "end_line": 13},
        "new": {"start_line": 10, "end_line": 13},
    }


def test_pure_insertion_has_anchor_and_no_old_edit_range():
    patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -3,0 +3,2 @@
+x
+y
"""
    result = parse_unified_patch(patch)["files"][0]
    assert result["old_edit_ranges"] == []
    assert result["new_edit_ranges"] == [{"start_line": 3, "end_line": 4}]
    assert result["insertion_anchors"] == [3]
    assert result["hunk_ranges"][0]["old"] is None


def test_parser_supports_quoted_rename_and_dev_null():
    patch = """diff --git \"a/old name.py\" \"b/new name.py\"
--- \"a/old name.py\"
+++ \"b/new name.py\"
@@ -1 +1 @@
-old
+new
diff --git a/new.py b/new.py
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+x
"""
    files = parse_unified_patch(patch)["files"]
    assert files[0]["old_path"] == "old name.py"
    assert files[0]["new_path"] == "new name.py"
    assert files[0]["status"] == "renamed"
    assert files[1]["old_path"] is None
    assert files[1]["new_path"] == "new.py"
    assert files[1]["status"] == "added"

    deleted = parse_unified_patch(
        "diff --git a/deleted.py b/deleted.py\n"
        "--- a/deleted.py\n+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n-x\n"
    )["files"][0]
    assert deleted["old_path"] == "deleted.py"
    assert deleted["new_path"] is None
    assert deleted["status"] == "deleted"


def test_quoted_diff_header_preserves_spaces_without_file_headers():
    patch = r'''diff --git "a/old name.py" "b/new name.py"
similarity index 100%
rename from old name.py
rename to new name.py
'''
    result = parse_unified_patch(patch)["files"][0]
    assert result["old_path"] == "old name.py"
    assert result["new_path"] == "new name.py"


def test_old_parser_path_alias_remains_available():
    patch = "diff --git a/a.py b/a.py\n@@ -10,2 +10,3 @@\n-x\n+y"
    result = parse_unified_patch(patch)
    assert result["files"][0]["path"] == "a.py"
    assert result["files"][0]["modified_ranges"][0]["old_start"] == 10
