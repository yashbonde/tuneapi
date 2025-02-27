# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

import tuneapi.types as tt

from unittest import TestCase, main as ut_main


def get_tree() -> tt.ThreadsTree:
    # NEVER CHANGE THIS
    return tt.ThreadsTree(
        tt.system("You are Tune Blob"),
        [
            tt.human("Who are you?", id="msg_000"),
            [
                tt.assistant("I am Tune Blob, a bot created by Tune API"),
                [
                    tt.human("What tools do you have?"),
                    [
                        [
                            tt.assistant("I don't have any tools yet."),
                        ],
                        [
                            tt.assistant(
                                "```<function ... you go.",
                                fn_pairs=[
                                    (
                                        tt.function_call(
                                            {"name": "list_tools", "arguments": "{}"}
                                        ),
                                        tt.function_resp("{...}"),
                                    )
                                ],
                            ),
                            [tt.human("What is 2 + 2?"), [tt.assistant("4")]],
                        ],
                    ],
                ],
            ],
            [
                tt.assistant(
                    "I am Tune Blob, a bot created by Tune AI. The super tech company!"
                ),
                [
                    [
                        tt.human("Send email to Mr. Steve Jobs ..."),
                        [
                            tt.assistant(
                                "```<function ... @apple.com] sent!",
                                fn_pairs=[
                                    (
                                        tt.function_call(
                                            {"name": "send_email", "arguments": "{}"}
                                        ),
                                        tt.function_resp("{...}"),
                                    )
                                ],
                            )
                        ],
                    ],
                    [
                        tt.human("Call the trinklet man ...", id="msg_010"),
                        [
                            [
                                tt.assistant("Sorry I ... thank you."),
                            ],
                            [tt.assistant("I Apolo ... take care", id="msg_100")],
                        ],
                    ],
                ],
            ],
        ],
    )


class Test_ThreadsTree(TestCase):
    def test_01_ser_deser(self):
        # ser/deser
        ttree = get_tree()
        clone = tt.ThreadsTree.from_dict(ttree.to_dict())

        self.assertEqual(
            clone.tree.diff(ttree.tree, reduce=True).count, 0, "Undo failed"
        )

    def test_02_add_undo(self):
        ttree = get_tree()
        clone = tt.ThreadsTree.from_dict(ttree.to_dict())

        # 2 adds
        ttree.add(tt.human("What is 2 + 2?"))
        ttree.undo()

        self.assertEqual(
            clone.tree.diff(ttree.tree, reduce=True).count, 0, "Undo failed"
        )

    def test_03_consecutive_same_role_pass(self):
        ttree = get_tree()
        ttree.add(tt.human("What is 2 + 2?"))
        ttree.add(tt.assistant("It's 4"))

    def test_04_consecutive_same_role_fail(self):
        ttree = get_tree()
        ttree.add(tt.human("What is 2 + 2?"))
        self.assertRaises(ValueError, ttree.add, tt.human("It's 4"))

    def test_05_insert_impossible(self):
        ttree = get_tree()
        self.assertRaises(ValueError, ttree.add, tt.human("What is 2 + 2?"), 101)

    def test_06_delete_impossible(self):
        ttree = get_tree()
        self.assertRaises(ValueError, ttree.delete, 101)

    def test_07_add_delete_parent(self):
        ttree = get_tree()
        ttree.add(tt.human("What is 2 + 2?", id="msg_001"))
        self.assertIsNotNone(ttree["msg_001"])
        ttree.delete("msg_000")
        self.assertRaises(ValueError, ttree.__getitem__, "msg_001")

    def test_08_regenerate_gpt(self):
        ttree = get_tree()
        ttree.regenerate(None, "msg_100", dry=True)

    def test_09_regenerate_human(self):
        ttree = get_tree()
        ttree.regenerate(None, "msg_010", "Hula", dry=True)

    def test_10_regenerate_human_fail_no_prompt(self):
        ttree = get_tree()
        self.assertRaises(ValueError, ttree.regenerate, None, "msg_010")

    def test_11_property_latest_message(self):
        ttree = get_tree()
        self.assertEqual(ttree.latest_message.id, "msg_100")

    def test_12_property_size(self):
        ttree = get_tree()
        self.assertEqual(ttree.size, 13)

    def test_13_property_degree_of_tree(self):
        ttree = get_tree()
        self.assertEqual(ttree.degree_of_tree, 2)


class Test_Thread(TestCase):
    def test_01_ser_deser(self):
        thread = get_tree().pick()
        tt.Thread.from_dict(thread.to_dict())

    def test_02_copy_paste(self):
        thread = get_tree().pick()
        thread.copy()


if __name__ == "__main__":
    ut_main()
