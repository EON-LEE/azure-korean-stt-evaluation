import unicodedata

from kstt_eval.normalize_ko import has_kspon_markup, normalize, remove_spaces


def test_dual_transcription_picks_one_side():
    text = "(그래서)/(그래가지고) 갔어요."
    assert normalize(text) == "그래서 갔어요"
    assert normalize(text, dual="pronunciation") == "그래가지고 갔어요"


def test_noise_tags_and_markers_removed():
    assert normalize("b/ 안녕하세요 n/") == "안녕하세요"
    assert normalize("어/ 그러니까+ 좋아요*") == "어 그러니까 좋아요"


def test_tag_letters_inside_words_are_kept():
    assert normalize("url/경로") == "url 경로"


def test_case_punctuation_whitespace_and_nfc():
    assert normalize("  AI   모델, 정말?!  ") == "ai 모델 정말"
    assert normalize(unicodedata.normalize("NFD", "한국어")) == "한국어"
    assert normalize(None) == ""
    assert normalize("") == ""


def test_markup_detection_and_space_removal():
    assert has_kspon_markup("(a)/(b)")
    assert has_kspon_markup("o/ 네")
    assert not has_kspon_markup("일반 문장")
    assert remove_spaces("가 나  다") == "가나다"
