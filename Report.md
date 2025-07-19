# CS336 Spring 2025 Assignment 1: Basics

## Problem (unicode1): Understanding Unicode (1 point)

(a) `chr(0) --> U+0000`，为空字符。

(b) `chr(0).__repr__()` 显示 `'\x00'`, `print` 时没有任何显式输出。

(c) 当嵌入文本时，它仍然是字符串的一部分（影响长度），但在 `print` 时是不可见的。

## Problem (unicode2): Unicode Encodings (3 points)

(a) 因为 UTF-8 编码兼容 ASCII 且在常见语言中更紧凑，无须为每个字符填充额外的字节，而 UTF-16/32 会引入大量零字节并占用更多存储空间，不利于分词器学习。

- UTF-32 中，单个英文字符需要占据 4 个字节。
- UTF-16 中，每个字符根据其对应的码位（code point）大小，可以使用 2 个字节表示或者 4 个字节表示。
- UTF-8 中，每个字符根据其对应的码位（code point）大小，可以使用 1 个或者 2 个或者 4 个字节表示，

(b) 以 `test_string="hello! こんにちは!"` 为例，日文 UTF-8 编码并非 1:1 匹配，该函数逐字节解码会出错。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python-input-0>", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
```

(c) 例如 `b'\xff\x80'`、`b'\xc0\x00'` 在 UTF-8 中并非任何合法字符的编码，因此无法解码成有效的 Unicode 字符。

```python
>>> (b'\xff\x80').decode("utf-8")
Traceback (most recent call last):
  File "<python-input-2>", line 1, in <module>
    (b'\xff\x80').decode("utf-8")
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

在 UTF-8 中，
- 如果第一个字节 (leading byte) 的最高位是 0，那么表示占 1 个字节（兼容 ASCII）。
- 如果第一个字节的最高三位是 110，那么表示这个字符占 2 个字节，第二个字节的最高 2 位是 10。
- 如果第一个字节的最高四位是 1110，那么表示这个字符占 3 个字节，第 2 和第 3 个字节的最高 2 位都是 10。
- 如果第一个字节的最高五位是 11110，那么表示这个字符占 4 个字节，第 2 和第 3 个和第 4 个字节的最高 2 位都是 10。
- UTF-8 还可以扩展到 6 字节，这样就能表示更多的码位，

```
1100 0000, 1000 0000
c 0, 8 0 
```

然而 `\xc0\x81` 虽然字节格式合法，但它是对 `U+0001` 的超长编码，根据 UTF-8 规范同样被禁止。

