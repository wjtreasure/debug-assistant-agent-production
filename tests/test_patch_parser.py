from debug_assistant.datasets.patch_parser import parse_unified_patch

def test_parse_patch():
    p="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -10,2 +10,3 @@\n-x\n+y"
    r=parse_unified_patch(p)
    assert r['files'][0]['path']=='a.py'
    assert r['files'][0]['modified_ranges'][0]['old_start']==10
