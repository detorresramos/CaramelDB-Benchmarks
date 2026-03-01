package com.randorithms.app;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.GZIPInputStream;

import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPException;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Parameter;
import com.martiansoftware.jsap.SimpleJSAP;
import com.martiansoftware.jsap.Switch;
import com.martiansoftware.jsap.UnflaggedOption;
import com.martiansoftware.jsap.stringparsers.FileStringParser;
import com.martiansoftware.jsap.stringparsers.ForNameStringParser;

import it.unimi.dsi.Util; // not sure
// import it.unimi.dsi.bits.BitVector;
// import it.unimi.dsi.bits.BitVectors;
// import it.unimi.dsi.bits.LongArrayBitVector;
// import it.unimi.dsi.bits.LongBigArrayBitVector;
import it.unimi.dsi.bits.TransformationStrategies;
import it.unimi.dsi.bits.TransformationStrategy;
// import it.unimi.dsi.fastutil.Size64;
import it.unimi.dsi.fastutil.io.BinIO;
// import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
// import it.unimi.dsi.fastutil.longs.LongBigList;
import it.unimi.dsi.fastutil.longs.LongIterable;
// import it.unimi.dsi.fastutil.longs.LongList;
// import it.unimi.dsi.fastutil.objects.AbstractObject2LongFunction;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.io.FileLinesByteArrayIterable;
import it.unimi.dsi.io.FileLinesMutableStringIterable;
// import it.unimi.dsi.io.OfflineIterable;
// import it.unimi.dsi.io.OfflineIterable.OfflineIterator;
import it.unimi.dsi.logging.ProgressLogger; // not sure
import it.unimi.dsi.sux4j.mph.codec.Codec;
// import it.unimi.dsi.sux4j.mph.codec.Codec.Decoder;
// import it.unimi.dsi.sux4j.mph.codec.Codec.Huffman;
// import it.unimi.dsi.sux4j.mph.codec.Codec.ZeroCodec;
import it.unimi.dsi.sux4j.mph.GV3CompressedFunction;


public class App 
{
	public static void main(final String[] arg) throws NoSuchMethodException, IOException, JSAPException {

		final SimpleJSAP jsap = new SimpleJSAP(GV3CompressedFunction.class.getName(), "Builds a GOV function mapping a newline-separated list" + " of strings to their ordinal position, or to specific values.", new Parameter[] {
				new FlaggedOption("encoding", ForNameStringParser.getParser(Charset.class), "UTF-8", JSAP.NOT_REQUIRED, 'e', "encoding", "The string file encoding."),
				new FlaggedOption("tempDir", FileStringParser.getParser(), JSAP.NO_DEFAULT, JSAP.NOT_REQUIRED, 'T', "temp-dir", "A directory for temporary files."),
				new Switch("iso", 'i', "iso", "Use ISO-8859-1 coding internally (i.e., just use the lower eight bits of each character)."),
				new Switch("peel", 'p', "peel", "Use peeling instead of lazy Gaussian elimination (+12% space, much faster construction)."),
				new Switch("utf32", JSAP.NO_SHORTFLAG, "utf-32", "Use UTF-32 internally (handles surrogate pairs)."),
				new Switch("byteArray", 'b', "byte-array", "Create a function on byte arrays (no character encoding)."),
				new Switch("zipped", 'z', "zipped", "The string list is compressed in gzip format."),
				new FlaggedOption("decompressor", JSAP.CLASS_PARSER, JSAP.NO_DEFAULT, JSAP.NOT_REQUIRED, 'd', "decompressor", "Use this extension of InputStream to decompress the strings (e.g., java.util.zip.GZIPInputStream)."),
				new FlaggedOption("codec", JSAP.STRING_PARSER, "HUFFMAN", JSAP.NOT_REQUIRED, 'C', "codec", "The name of the codec to use (UNARY, BINARY, GAMMA, HUFFMAN, LLHUFFMAN)."),
				new FlaggedOption("limit", JSAP.INTEGER_PARSER, "20", JSAP.NOT_REQUIRED, 'l', "limit", "Decoding-table length limit for the LLHUFFMAN codec."),
				new FlaggedOption("values", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.NOT_REQUIRED, 'v', "values", "A binary file in DataInput format containing a long for each string (otherwise, the values will be the ordinal positions of the strings)."),
				new UnflaggedOption("function", JSAP.STRING_PARSER, JSAP.NO_DEFAULT, JSAP.REQUIRED, JSAP.NOT_GREEDY, "The filename for the serialised GOV function."),
				new UnflaggedOption("stringFile", JSAP.STRING_PARSER, "-", JSAP.NOT_REQUIRED, JSAP.NOT_GREEDY, "The name of a file containing a newline-separated list of strings, or - for standard input; in the second case, strings must be fewer than 2^31 and will be loaded into core memory."), });

		final JSAPResult jsapResult = jsap.parse(arg);
		if (jsap.messagePrinted()) return;

		final String functionName = jsapResult.getString("function");
		final String stringFile = jsapResult.getString("stringFile");
		final Charset encoding = (Charset)jsapResult.getObject("encoding");
		final File tempDir = jsapResult.getFile("tempDir");
		final boolean byteArray = jsapResult.getBoolean("byteArray");
		final boolean zipped = jsapResult.getBoolean("zipped");
		Class<? extends InputStream> decompressor = jsapResult.getClass("decompressor");
		final boolean peeled = jsapResult.getBoolean("peel");
		final boolean iso = jsapResult.getBoolean("iso");
		final boolean utf32 = jsapResult.getBoolean("utf32");
		final int limit = jsapResult.getInt("limit");

		if (zipped && decompressor != null) throw new IllegalArgumentException("The zipped and decompressor options are incompatible");
		if (zipped) decompressor = GZIPInputStream.class;

		Codec codec = null;
		switch (jsapResult.getString("codec")) {
		case "UNARY":
			codec = new Codec.Unary();
			break;
		case "BINARY":
			codec = new Codec.Binary();
			break;
		case "GAMMA":
			codec = new Codec.Gamma();
			break;
		case "HUFFMAN":
			codec = new Codec.Huffman();
			break;
		case "LLHUFFMAN":
			codec = new Codec.Huffman(limit);
			break;
		default:
			throw new IllegalArgumentException("Unknown codec \"" + jsapResult.getString("codec") + "\"");
		}

		final LongIterable values = jsapResult.userSpecified("values") ? BinIO.asLongIterable(jsapResult.getString("values")) : null;

		if (byteArray) {
			if ("-".equals(stringFile)) throw new IllegalArgumentException("Cannot read from standard input when building byte-array functions");
			if (iso || utf32 || jsapResult.userSpecified("encoding")) throw new IllegalArgumentException("Encoding options are not available when building byte-array functions");
			final Iterable<byte[]> keys = new FileLinesByteArrayIterable(stringFile, decompressor);
			GV3CompressedFunction<byte[]> csf = null;
			GV3CompressedFunction.Builder<byte[]> builder = new GV3CompressedFunction.Builder<byte[]>()
				.keys(keys)
                .transform(TransformationStrategies.rawByteArray())
				.values(values)
				.tempDir(tempDir)
				.codec(codec);
			if (peeled){
				builder.peeled();
			}
			csf = builder.build();
			// new GV3CompressedFunction<>(keys, TransformationStrategies.rawByteArray(), values, false, tempDir, null, codec, peeled)
			// keys, transform, values, indirect, tempdir, bucketedhashstore, codec, peeled
			System.out.println("Size of function: " + csf.numBits() + " bits.");
			BinIO.storeObject(csf, functionName);
		} else {
			final Iterable<? extends CharSequence> keys;
			if ("-".equals(stringFile)) {
				final ObjectArrayList<String> list = new ObjectArrayList<>();
				keys = list;
				FileLinesMutableStringIterable.iterator(System.in, encoding, decompressor).forEachRemaining(s -> list.add(s.toString()));
			} else keys = new FileLinesMutableStringIterable(stringFile, encoding, decompressor);
			final TransformationStrategy<CharSequence> transformationStrategy = iso ? TransformationStrategies.rawIso() : utf32 ? TransformationStrategies.rawUtf32() : TransformationStrategies.rawUtf16();

			GV3CompressedFunction<CharSequence> csf = null;
			GV3CompressedFunction.Builder<CharSequence> builder = new GV3CompressedFunction.Builder<CharSequence>()
				.keys(keys)
                .transform(transformationStrategy)
				.values(values)
				.tempDir(tempDir)
				.codec(codec);
			if (peeled){
				builder.peeled();
			}
			csf = builder.build();
			System.out.println("Size of function: " + csf.numBits() + " bits.");
			// keys, transform, values, indirect, tempdir, bucketedhashstore, codec, peeled
			BinIO.storeObject(csf, functionName);
		}
    }
}


// mph = new GOVMinimalPerfectHashFunction.Builder<K>()
//                 .transform(transformationStrategy)
//                 .signed(config.getSignatureWidth())
//                 .keys(trackMinMaxKeys ? new PairFirstRangeTrackingIterable(entries, minMaxKeys)
//                       : new PairFirstIterable(entries))
//                 .build();

// csf = new GV3CompressedFunction.Builder<K>()
// 	.transform(transformationStrategy)
// 	.keys(keys)
// 	.values(values)
// 	.build();

// csf = new GV3CompressedFunction.Builder<byte[]>()
// 	.keys(keys)
//     .transform(TransformationStrategies.rawByteArray())
// 	.values(values)
// 	.tempDir(tempDir)
// 	.codec(codec)
// 	.peeled(peeled)
//     .build();

