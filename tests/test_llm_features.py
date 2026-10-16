import random
from src.features import shuffle_quiz_item, validate_quiz_item
from src.llm import parse_json

def test_parse_json_fence(): assert parse_json('```json\n{"a":1}\n```')=={"a":1}
def test_quiz_validation():
    good={"question":"Q?","options":["a","b","c","d"],"answer_index":2,"explanation":"Because"}
    assert validate_quiz_item(good); assert not validate_quiz_item({"question":"x"})
def test_shuffle_preserves_correct():
    item={"question":"Q?","options":["a","b","c","d"],"answer_index":2,"explanation":"Because"}
    out=shuffle_quiz_item(item,random.Random(3)); assert out["options"][out["answer_index"]]=="c"
